//! Angular locality-sensitive hash families over a seeded random rotation.
//!
//! - `hyperplane`: one rotation per table; bit `j` of the key is 1 when rotated coordinate `j`
//!   exceeds 1e-12 times the vector's norm, for `j < k`. The k hyperplanes of a table are rows of
//!   one orthogonal matrix.
//! - `cross_polytope` (Andoni et al. 2015): k rotations per table; each hash is `2 * i + s`, where
//!   `i` is the index of the largest-magnitude rotated coordinate (lowest index on ties) and `s` is 1
//!   when that coordinate is negative. The table key is `sum_j h_j * (2n)^j` for padded length `n`.
//!
//! Rotation seeds are `splitmix64(seed, table * k + j)`. Both families depend only on the direction
//! of a vector, so any positive scale gives the same keys.
//!
//! Multi-probe: `probes` writes the vector's own key first, then the keys a near neighbour is most
//! likely to have instead. Hyperplane flips one bit, smallest |coordinate| first. Cross-polytope
//! replaces one hash by a runner-up coordinate, smallest gap `|y_max| - |y_i|` first. Magnitudes are
//! compared on a grid of 1e-9 times the vector's norm, so exact ties stay ties; ties go to the lower
//! coordinate for maxima and runner-ups, and to the lower hash position, then rank, for probes.

const std = @import("std");
const rotate = @import("rotate.zig");

pub const Family = enum { hyperplane, cross_polytope };

pub const Params = struct {
    family: Family,
    /// Bits per table (hyperplane) or hashes per table (cross-polytope).
    k: u8,
    tables: u32,
    seed: u64,
};

pub const Error = error{
    DimsZero,
    KOutOfRange,
    NoTables,
    TooManyProbes,
    DimensionMismatch,
    ScratchTooSmall,
    NonFiniteValue,
    TableOutOfRange,
} || rotate.Error;

/// Largest number of probes per table, own key included.
pub const max_probes = 65;

/// Bits of a cross-polytope key used by one hash: `log2(2 * padded)`.
fn bitsPerHash(padded: usize) usize {
    return @as(usize, std.math.log2_int(usize, padded)) + 1;
}

/// The most probe alternatives any valid `Params` can produce in one table.
const max_alts = blk: {
    const max_pow2: usize = @as(usize, 1) << (@bitSizeOf(usize) - 1);
    var most: usize = 64;
    var padded: usize = 2;
    while (true) : (padded *= 2) {
        const k = 64 / bitsPerHash(padded);
        most = @max(most, k * @min(max_probes - 1, padded - 1));
        if (padded == max_pow2) break;
    }
    break :blk most;
};

pub fn validate(p: Params, dims: usize) Error!void {
    if (dims == 0) return error.DimsZero;
    if (p.tables == 0) return error.NoTables;
    const padded = try rotate.paddedLen(dims);
    switch (p.family) {
        .hyperplane => if (p.k == 0 or p.k > 64 or p.k > padded) return error.KOutOfRange,
        .cross_polytope => {
            if (p.k == 0 or @as(usize, p.k) * bitsPerHash(padded) > 64) return error.KOutOfRange;
        },
    }
}

pub fn rotationSeed(p: Params, table: u32, j: usize) u64 {
    return rotate.splitmix64(p.seed, @as(u64, table) * p.k + j);
}

/// `key` and `probes` write into the hasher's own buffer, so one `Hasher` must not be used from
/// several threads at once. `keyWith` and `probesWith` only read the hasher and write into a
/// caller-supplied scratch buffer of `scratchLen()` values instead.
pub const Hasher = struct {
    params: Params,
    dims: usize,
    padded: usize,
    /// k rotated copies of the input, `k * padded` values.
    buf: []f64,

    pub fn init(gpa: std.mem.Allocator, dims: usize, params: Params) (Error || std.mem.Allocator.Error)!Hasher {
        try validate(params, dims);
        const padded = try rotate.paddedLen(dims);
        const len = std.math.mul(usize, rowsOf(params), padded) catch return error.OutOfMemory;
        return .{ .params = params, .dims = dims, .padded = padded, .buf = try gpa.alloc(f64, len) };
    }

    pub fn deinit(self: Hasher, gpa: std.mem.Allocator) void {
        gpa.free(self.buf);
    }

    fn rowsOf(p: Params) usize {
        return if (p.family == .hyperplane) 1 else p.k;
    }

    /// Values of scratch that `keyWith` and `probesWith` need.
    pub fn scratchLen(self: Hasher) usize {
        return rowsOf(self.params) * self.padded;
    }

    /// The hasher with its buffer replaced by `scratch`, after checking `x`, `table`, and the fields.
    fn bind(self: Hasher, scratch: []f64, x: []const f32, table: u32) Error!Hasher {
        try validate(self.params, self.dims);
        if (table >= self.params.tables) return error.TableOutOfRange;
        if (self.padded != try rotate.paddedLen(self.dims)) return error.DimensionMismatch;
        if (x.len != self.dims) return error.DimensionMismatch;
        for (x) |v| if (!std.math.isFinite(v)) return error.NonFiniteValue;
        const needed = std.math.mul(usize, rowsOf(self.params), self.padded) catch return error.DimensionTooLarge;
        if (scratch.len < needed) return error.ScratchTooSmall;
        var h = self;
        h.buf = scratch[0..needed];
        return h;
    }

    fn row(self: Hasher, j: usize) []f64 {
        return self.buf[j * self.padded ..][0..self.padded];
    }

    fn fill(self: Hasher, x: []const f32, table: u32) Error!void {
        for (0..rowsOf(self.params)) |j| try rotate.apply(rotationSeed(self.params, table, j), x, self.row(j));
    }

    /// |v| on a grid of 1e-9 times the vector's norm, so that magnitudes equal in exact
    /// arithmetic compare equal whatever the rounding (padding makes such ties common).
    fn mag(v: f64, inv_unit: f64) f64 {
        return @floor(@abs(v) * inv_unit + 0.5);
    }

    fn invUnitOf(y: []const f64) f64 {
        var s: f64 = 0;
        for (y) |v| s += v * v;
        return if (s == 0) 1 else 1.0 / (1e-9 * @sqrt(s));
    }

    fn argmaxWithInvUnit(y: []const f64, inv_u: f64) usize {
        var best: usize = 0;
        var best_mag = mag(y[0], inv_u);
        for (y[1..], 1..) |v, i| {
            const mv = mag(v, inv_u);
            if (mv > best_mag) {
                best = i;
                best_mag = mv;
            }
        }
        return best;
    }

    fn argmax(y: []const f64) usize {
        return argmaxWithInvUnit(y, invUnitOf(y));
    }

    fn cpHash(y: []const f64, i: usize) u64 {
        return 2 * @as(u64, i) + @intFromBool(y[i] < 0);
    }

    fn compose(self: Hasher, hashes: []const u64) u64 {
        var acc: u64 = 0;
        var mul: u64 = 1;
        for (hashes) |h| {
            acc +%= h *% mul;
            mul *%= 2 *% @as(u64, self.padded);
        }
        return acc;
    }

    /// Rotated coordinates at most this large count as zero for hyperplane bits. With padding, a
    /// rotation row can be exactly zero on the input's coordinates; its sign is then rounding noise.
    fn zeroThreshold(self: Hasher) f64 {
        var s: f64 = 0;
        for (self.row(0)) |v| s += v * v;
        return 1e-12 * @sqrt(s);
    }

    /// The key of `x` in `table`, using the hasher's own buffer.
    pub fn key(self: *Hasher, x: []const f32, table: u32) Error!u64 {
        return self.keyWith(self.buf, x, table);
    }

    /// The key of `x` in `table`, using `scratch` (at least `scratchLen()` values) as the buffer.
    pub fn keyWith(self: Hasher, scratch: []f64, x: []const f32, table: u32) Error!u64 {
        const h = try self.bind(scratch, x, table);
        try h.fill(x, table);
        return h.keyFilled();
    }

    fn keyFilled(self: Hasher) u64 {
        const p = self.params;
        switch (p.family) {
            .hyperplane => {
                const thr = self.zeroThreshold();
                var k: u64 = 0;
                for (self.row(0)[0..p.k], 0..) |v, j| {
                    if (v > thr) k |= @as(u64, 1) << @intCast(j);
                }
                return k;
            },
            .cross_polytope => {
                var hashes: [64]u64 = undefined;
                for (0..p.k) |j| hashes[j] = cpHash(self.row(j), argmax(self.row(j)));
                return self.compose(hashes[0..p.k]);
            },
        }
    }

    const Alt = struct { gap: f64, coord: usize, pos: u8, rank: u8 };

    fn altLess(_: void, a: Alt, b: Alt) bool {
        const a_nan = std.math.isNan(a.gap);
        const b_nan = std.math.isNan(b.gap);
        if (a_nan != b_nan) return !a_nan;
        if (!a_nan and a.gap != b.gap) return a.gap < b.gap;
        if (a.pos != b.pos) return a.pos < b.pos;
        return a.rank < b.rank;
    }

    /// Writes the own key and then up to `out.len - 1` probe keys of `x` in `table`; returns the
    /// number written (fewer when the family has fewer alternatives). Uses the hasher's own buffer.
    pub fn probes(self: *Hasher, x: []const f32, table: u32, out: []u64) Error!usize {
        return self.probesWith(self.buf, x, table, out);
    }

    /// As `probes`, using `scratch` (at least `scratchLen()` values) as the buffer.
    pub fn probesWith(self: Hasher, scratch: []f64, x: []const f32, table: u32, out: []u64) Error!usize {
        if (out.len > max_probes) return error.TooManyProbes;
        const h = try self.bind(scratch, x, table);
        if (out.len == 0) return 0;
        return h.probesFilled(x, table, out);
    }

    fn probesFilled(self: Hasher, x: []const f32, table: u32, out: []u64) Error!usize {
        try self.fill(x, table);
        const want = out.len - 1;
        if (want == 0) {
            out[0] = self.keyFilled();
            return 1;
        }
        const p = self.params;
        var alts: [max_alts]Alt = undefined;
        var base_hashes: [64]u64 = undefined;
        var count: usize = 0;
        switch (p.family) {
            .hyperplane => {
                const y = self.row(0);
                var s: f64 = 0;
                for (y) |v| s += v * v;
                const norm = @sqrt(s);
                const thr = 1e-12 * norm;
                const inv_u: f64 = if (s == 0) 1 else 1.0 / (1e-9 * norm);
                var own: u64 = 0;
                for (y[0..p.k], 0..) |v, j| {
                    if (v > thr) own |= @as(u64, 1) << @intCast(j);
                    alts[count] = .{ .gap = mag(v, inv_u), .pos = @intCast(j), .rank = 0, .coord = j };
                    count += 1;
                }
                out[0] = own;
            },
            .cross_polytope => {
                for (0..p.k) |j| {
                    const y = self.row(j);
                    const inv_u = invUnitOf(y);
                    const keep = @min(want + 1, y.len);
                    var top_mags: [max_probes]f64 = undefined;
                    var top_coords: [max_probes]usize = undefined;
                    var len: usize = 0;
                    for (y, 0..) |v, i| {
                        const mv = mag(v, inv_u);
                        if (len == keep and mv <= top_mags[keep - 1]) continue;
                        var pos = if (len < keep) len else keep - 1;
                        while (pos > 0 and mv > top_mags[pos - 1]) : (pos -= 1) {
                            top_mags[pos] = top_mags[pos - 1];
                            top_coords[pos] = top_coords[pos - 1];
                        }
                        top_mags[pos] = mv;
                        top_coords[pos] = i;
                        if (len < keep) len += 1;
                    }
                    const top = top_coords[0];
                    const top_mag = top_mags[0];
                    base_hashes[j] = cpHash(y, top);
                    for (1..len) |idx| {
                        alts[count] = .{
                            .gap = top_mag - top_mags[idx],
                            .pos = @intCast(j),
                            .rank = @intCast(idx - 1),
                            .coord = top_coords[idx],
                        };
                        count += 1;
                    }
                }
                out[0] = self.compose(base_hashes[0..p.k]);
            },
        }
        std.mem.sort(Alt, alts[0..count], {}, altLess);
        const n = @min(want, count);
        var hashes: [64]u64 = undefined;
        if (p.family == .cross_polytope) {
            @memcpy(hashes[0..p.k], base_hashes[0..p.k]);
        }
        for (alts[0..n], 1..) |a, o| {
            switch (p.family) {
                .hyperplane => out[o] = out[0] ^ (@as(u64, 1) << @intCast(a.coord)),
                .cross_polytope => {
                    hashes[a.pos] = cpHash(self.row(a.pos), a.coord);
                    out[o] = self.compose(hashes[0..p.k]);
                    hashes[a.pos] = base_hashes[a.pos];
                },
            }
        }
        return n + 1;
    }
};

test "keys ignore positive scale" {
    const gpa = std.testing.allocator;
    const x = [_]f32{ 0.3, -1.2, 0.8, 0.05, -0.4 };
    var y: [5]f32 = undefined;
    for (&y, x) |*a, b| a.* = 3.5 * b;
    inline for (.{ Family.hyperplane, Family.cross_polytope }) |f| {
        var h = try Hasher.init(gpa, 5, .{ .family = f, .k = 2, .tables = 3, .seed = 11 });
        defer h.deinit(gpa);
        for (0..3) |t| try std.testing.expectEqual(try h.key(&x, @intCast(t)), try h.key(&y, @intCast(t)));
    }
}

test "negation: hyperplane complements, cross-polytope flips every sign" {
    const gpa = std.testing.allocator;
    const x = [_]f32{ 0.3, -1.2, 0.8, 0.05, -0.4, 0.9, 0.2 };
    var nx: [7]f32 = undefined;
    for (&nx, x) |*a, b| a.* = -b;
    var hp = try Hasher.init(gpa, 7, .{ .family = .hyperplane, .k = 8, .tables = 1, .seed = 5 });
    defer hp.deinit(gpa);
    try std.testing.expectEqual(try hp.key(&x, 0) ^ 0xff, try hp.key(&nx, 0));
    var cp = try Hasher.init(gpa, 7, .{ .family = .cross_polytope, .k = 1, .tables = 1, .seed = 5 });
    defer cp.deinit(gpa);
    try std.testing.expectEqual(try cp.key(&x, 0) ^ 1, try cp.key(&nx, 0));
}

test "probes start with the own key and are distinct" {
    const gpa = std.testing.allocator;
    const x = [_]f32{ 0.3, -1.2, 0.8, 0.05, -0.4, 0.9, 0.2, -0.7, 1.1 };
    inline for (.{ Family.hyperplane, Family.cross_polytope }) |f| {
        var h = try Hasher.init(gpa, 9, .{ .family = f, .k = 2, .tables = 2, .seed = 3 });
        defer h.deinit(gpa);
        var out: [6]u64 = undefined;
        const n = try h.probes(&x, 1, &out);
        try std.testing.expectEqual(@as(usize, if (f == .hyperplane) 3 else 6), n);
        try std.testing.expectEqual(try h.key(&x, 1), out[0]);
        for (out[0..n], 0..) |a, i| for (out[i + 1 .. n]) |b| try std.testing.expect(a != b);
    }
}

test "validation" {
    try std.testing.expectError(error.DimsZero, validate(.{ .family = .hyperplane, .k = 8, .tables = 1, .seed = 0 }, 0));
    try std.testing.expectError(error.NoTables, validate(.{ .family = .hyperplane, .k = 8, .tables = 0, .seed = 0 }, 4));
    try std.testing.expectError(error.KOutOfRange, validate(.{ .family = .hyperplane, .k = 65, .tables = 1, .seed = 0 }, 128));
    try std.testing.expectError(error.KOutOfRange, validate(.{ .family = .cross_polytope, .k = 6, .tables = 1, .seed = 0 }, 1024));
    try validate(.{ .family = .cross_polytope, .k = 5, .tables = 1, .seed = 0 }, 1024);
}

test "validate rejects dimensions whose padding overflows" {
    const max_pow2: usize = @as(usize, 1) << (@bitSizeOf(usize) - 1);
    try std.testing.expectError(error.DimensionTooLarge, validate(.{ .family = .hyperplane, .k = 8, .tables = 1, .seed = 0 }, max_pow2 + 1));
    try validate(.{ .family = .cross_polytope, .k = 1, .tables = 1, .seed = 0 }, max_pow2);
}

test "key and probes reject a vector of the wrong width" {
    const gpa = std.testing.allocator;
    inline for (.{ Family.hyperplane, Family.cross_polytope }) |f| {
        var h = try Hasher.init(gpa, 4, .{ .family = f, .k = 2, .tables = 1, .seed = 3 });
        defer h.deinit(gpa);
        var out: [3]u64 = undefined;
        try std.testing.expectError(error.DimensionMismatch, h.key(&.{ 1, 2, 3 }, 0));
        try std.testing.expectError(error.DimensionMismatch, h.key(&.{ 1, 2, 3, 4, 5 }, 0));
        try std.testing.expectError(error.DimensionMismatch, h.probes(&.{ 1, 2, 3 }, 0, &out));
        var scratch: [16]f64 = undefined;
        try std.testing.expectError(error.ScratchTooSmall, h.keyWith(scratch[0 .. h.scratchLen() - 1], &.{ 1, 2, 3, 4 }, 0));
        try std.testing.expectError(error.ScratchTooSmall, h.probesWith(scratch[0 .. h.scratchLen() - 1], &.{ 1, 2, 3, 4 }, 0, &out));
        var bad = h;
        bad.params.k = 200;
        try std.testing.expectError(error.KOutOfRange, bad.keyWith(&scratch, &.{ 1, 2, 3, 4 }, 0));
    }
}

test "keyWith and probesWith match key and probes and leave the hasher untouched" {
    const gpa = std.testing.allocator;
    const x = [_]f32{ 0.3, -1.2, 0.8, 0.05, -0.4, 0.9, 0.2, -0.7, 1.1 };
    inline for (.{ Family.hyperplane, Family.cross_polytope }) |f| {
        var h = try Hasher.init(gpa, 9, .{ .family = f, .k = 2, .tables = 2, .seed = 3 });
        defer h.deinit(gpa);
        const scratch = try gpa.alloc(f64, h.scratchLen());
        defer gpa.free(scratch);
        var a: [max_probes]u64 = undefined;
        var b: [max_probes]u64 = undefined;
        for (0..2) |t| {
            @memset(h.buf, 0);
            const kw = try h.keyWith(scratch, &x, @intCast(t));
            const na = try h.probesWith(scratch, &x, @intCast(t), &a);
            for (h.buf) |v| try std.testing.expectEqual(@as(f64, 0), v);
            try std.testing.expectEqual(try h.key(&x, @intCast(t)), kw);
            const nb = try h.probes(&x, @intCast(t), &b);
            try std.testing.expectEqualSlices(u64, b[0..nb], a[0..na]);
        }
    }
}

test "the probe alternative buffer covers the largest valid table" {
    try std.testing.expectEqual(@as(usize, 567), max_alts);
    const gpa = std.testing.allocator;
    var x: [64]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast(i % 13)) - 6);
    var h = try Hasher.init(gpa, 64, .{ .family = .cross_polytope, .k = 9, .tables = 1, .seed = 8 });
    defer h.deinit(gpa);
    var out: [max_probes]u64 = undefined;
    try std.testing.expectEqual(@as(usize, max_probes), try h.probes(&x, 0, &out));
}

test "key and probes reject non-finite inputs and out-of-range table indices" {
    const gpa = std.testing.allocator;
    const bad_vecs = [_][4]f32{
        .{ 1, std.math.nan(f32), 0, 0 },
        .{ 1, std.math.inf(f32), 0, 0 },
        .{ -std.math.inf(f32), 0, 1, 2 },
    };
    inline for (.{ Family.hyperplane, Family.cross_polytope }) |f| {
        var h = try Hasher.init(gpa, 4, .{ .family = f, .k = 2, .tables = 2, .seed = 9 });
        defer h.deinit(gpa);
        var out: [4]u64 = undefined;
        for (bad_vecs) |v| {
            try std.testing.expectError(error.NonFiniteValue, h.key(&v, 0));
            try std.testing.expectError(error.NonFiniteValue, h.probes(&v, 0, &out));
            try std.testing.expectError(error.NonFiniteValue, h.probes(&v, 0, out[0..0]));
        }
        const ok = [_]f32{ 1, 2, 3, 4 };
        try std.testing.expectError(error.TableOutOfRange, h.key(&ok, 2));
        try std.testing.expectError(error.TableOutOfRange, h.key(&ok, std.math.maxInt(u32)));
        try std.testing.expectError(error.TableOutOfRange, h.probes(&ok, 2, &out));
        try std.testing.expectError(error.TableOutOfRange, h.probes(&ok, 2, out[0..0]));
    }
}

test "altLess is a total order even with non-finite gaps" {
    const nan = std.math.nan(f64);
    const inf = std.math.inf(f64);
    var items = [_]Hasher.Alt{
        .{ .gap = nan, .coord = 0, .pos = 1, .rank = 0 },
        .{ .gap = inf, .coord = 1, .pos = 0, .rank = 0 },
        .{ .gap = -inf, .coord = 2, .pos = 0, .rank = 0 },
        .{ .gap = 1.0, .coord = 3, .pos = 1, .rank = 0 },
        .{ .gap = nan, .coord = 4, .pos = 0, .rank = 1 },
        .{ .gap = 1.0, .coord = 5, .pos = 0, .rank = 2 },
        .{ .gap = nan, .coord = 6, .pos = 0, .rank = 0 },
    };
    for (items) |a| {
        try std.testing.expect(!Hasher.altLess({}, a, a));
        for (items) |b| {
            if (Hasher.altLess({}, a, b)) try std.testing.expect(!Hasher.altLess({}, b, a));
        }
    }
    std.mem.sort(Hasher.Alt, &items, {}, Hasher.altLess);
    try std.testing.expectEqual(@as(usize, 2), items[0].coord);
    try std.testing.expectEqual(@as(usize, 5), items[1].coord);
    try std.testing.expectEqual(@as(usize, 3), items[2].coord);
    try std.testing.expectEqual(@as(usize, 1), items[3].coord);
    try std.testing.expectEqual(@as(usize, 6), items[4].coord);
    try std.testing.expectEqual(@as(usize, 4), items[5].coord);
    try std.testing.expectEqual(@as(usize, 0), items[6].coord);
}

test "single-pass cross-polytope probes preserve tie order and zero-vector ties" {
    const gpa = std.testing.allocator;
    const zeros = [_]f32{ 0, 0, 0, 0, 0 };
    var h = try Hasher.init(gpa, 5, .{ .family = .cross_polytope, .k = 2, .tables = 1, .seed = 17 });
    defer h.deinit(gpa);
    var single: [1]u64 = undefined;
    try std.testing.expectEqual(@as(usize, 1), try h.probes(&zeros, 0, &single));
    try std.testing.expectEqual(try h.key(&zeros, 0), single[0]);
    var many: [15]u64 = undefined;
    const got = try h.probes(&zeros, 0, &many);
    try std.testing.expectEqual(@as(usize, 15), got);
    try std.testing.expectEqual(single[0], many[0]);
    for (many[0..got], 0..) |a, i| for (many[i + 1 .. got]) |b| try std.testing.expect(a != b);
}
