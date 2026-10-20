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

pub const Error = error{ DimsZero, KOutOfRange, NoTables, TooManyProbes };

/// Largest number of probes per table, own key included.
pub const max_probes = 65;

pub fn validate(p: Params, dims: usize) Error!void {
    if (dims == 0) return error.DimsZero;
    if (p.tables == 0) return error.NoTables;
    switch (p.family) {
        .hyperplane => if (p.k == 0 or p.k > 64 or p.k > rotate.paddedLen(dims)) return error.KOutOfRange,
        .cross_polytope => {
            const per = std.math.log2_int(usize, 2 * rotate.paddedLen(dims));
            if (p.k == 0 or @as(usize, p.k) * per > 64) return error.KOutOfRange;
        },
    }
}

pub fn rotationSeed(p: Params, table: u32, j: usize) u64 {
    return rotate.splitmix64(p.seed, @as(u64, table) * p.k + j);
}

pub const Hasher = struct {
    params: Params,
    dims: usize,
    padded: usize,
    /// k rotated copies of the input, `k * padded` values.
    buf: []f64,

    pub fn init(gpa: std.mem.Allocator, dims: usize, params: Params) !Hasher {
        try validate(params, dims);
        const padded = rotate.paddedLen(dims);
        const rows: usize = if (params.family == .hyperplane) 1 else params.k;
        return .{ .params = params, .dims = dims, .padded = padded, .buf = try gpa.alloc(f64, rows * padded) };
    }

    pub fn deinit(self: Hasher, gpa: std.mem.Allocator) void {
        gpa.free(self.buf);
    }

    fn row(self: Hasher, j: usize) []f64 {
        return self.buf[j * self.padded ..][0..self.padded];
    }

    fn fill(self: Hasher, x: []const f32, table: u32) void {
        std.debug.assert(x.len == self.dims);
        const rows: usize = if (self.params.family == .hyperplane) 1 else self.params.k;
        for (0..rows) |j| rotate.apply(rotationSeed(self.params, table, j), x, self.row(j));
    }

    /// |v| on a grid of 1e-9 times the vector's norm, so that magnitudes equal in exact
    /// arithmetic compare equal whatever the rounding (padding makes such ties common).
    fn mag(v: f64, unit: f64) f64 {
        return @floor(@abs(v) / unit + 0.5);
    }

    fn unitOf(y: []const f64) f64 {
        var s: f64 = 0;
        for (y) |v| s += v * v;
        return if (s == 0) 1 else 1e-9 * @sqrt(s);
    }

    fn argmaxWithUnit(y: []const f64, u: f64) usize {
        var best: usize = 0;
        var best_mag = mag(y[0], u);
        for (y[1..], 1..) |v, i| {
            const mv = mag(v, u);
            if (mv > best_mag) {
                best = i;
                best_mag = mv;
            }
        }
        return best;
    }

    fn argmax(y: []const f64) usize {
        return argmaxWithUnit(y, unitOf(y));
    }

    fn cpHash(y: []const f64, i: usize) u64 {
        return 2 * @as(u64, i) + @intFromBool(y[i] < 0);
    }

    fn compose(self: Hasher, hashes: []const u64) u64 {
        var acc: u64 = 0;
        var mul: u64 = 1;
        for (hashes) |h| {
            acc +%= h *% mul;
            mul *%= 2 * @as(u64, self.padded);
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

    /// The key of `x` in `table`.
    pub fn key(self: Hasher, x: []const f32, table: u32) u64 {
        self.fill(x, table);
        return self.keyFilled();
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

    const Alt = struct { gap: f64, pos: usize, rank: usize, coord: usize };

    fn altLess(_: void, a: Alt, b: Alt) bool {
        if (a.gap != b.gap) return a.gap < b.gap;
        if (a.pos != b.pos) return a.pos < b.pos;
        return a.rank < b.rank;
    }

    /// Writes the own key and then up to `out.len - 1` probe keys of `x` in `table`; returns the
    /// number written (fewer when the family has fewer alternatives).
    pub fn probes(self: Hasher, x: []const f32, table: u32, out: []u64) Error!usize {
        if (out.len == 0) return 0;
        if (out.len > max_probes) return error.TooManyProbes;
        self.fill(x, table);
        out[0] = self.keyFilled();
        const want = out.len - 1;
        if (want == 0) return 1;
        const p = self.params;
        var alts: [max_probes * 64]Alt = undefined;
        var base_hashes: [64]u64 = undefined;
        var count: usize = 0;
        switch (p.family) {
            .hyperplane => {
                const u = unitOf(self.row(0));
                for (self.row(0)[0..p.k], 0..) |v, j| {
                    alts[count] = .{ .gap = mag(v, u), .pos = j, .rank = 0, .coord = j };
                    count += 1;
                }
            },
            .cross_polytope => {
                for (0..p.k) |j| {
                    const y = self.row(j);
                    const u = unitOf(y);
                    const top = argmaxWithUnit(y, u);
                    base_hashes[j] = cpHash(y, top);
                    const top_mag = mag(y[top], u);
                    // The `want` largest |y_i| other than the maximum, found by repeated selection.
                    var taken: [max_probes]usize = undefined;
                    var r: usize = 0;
                    while (r < want and r + 1 < y.len) : (r += 1) {
                        var best: ?usize = null;
                        var best_mag: f64 = -1;
                        for (y, 0..) |v, i| {
                            if (i == top) continue;
                            if (std.mem.indexOfScalar(usize, taken[0..r], i) != null) continue;
                            const mv = mag(v, u);
                            if (best == null or mv > best_mag) {
                                best = i;
                                best_mag = mv;
                            }
                        }
                        taken[r] = best.?;
                        alts[count] = .{ .gap = top_mag - best_mag, .pos = j, .rank = r, .coord = best.? };
                        count += 1;
                    }
                }
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
        const h = try Hasher.init(gpa, 5, .{ .family = f, .k = 2, .tables = 3, .seed = 11 });
        defer h.deinit(gpa);
        for (0..3) |t| try std.testing.expectEqual(h.key(&x, @intCast(t)), h.key(&y, @intCast(t)));
    }
}

test "negation: hyperplane complements, cross-polytope flips every sign" {
    const gpa = std.testing.allocator;
    const x = [_]f32{ 0.3, -1.2, 0.8, 0.05, -0.4, 0.9, 0.2 };
    var nx: [7]f32 = undefined;
    for (&nx, x) |*a, b| a.* = -b;
    const hp = try Hasher.init(gpa, 7, .{ .family = .hyperplane, .k = 8, .tables = 1, .seed = 5 });
    defer hp.deinit(gpa);
    try std.testing.expectEqual(hp.key(&x, 0) ^ 0xff, hp.key(&nx, 0));
    const cp = try Hasher.init(gpa, 7, .{ .family = .cross_polytope, .k = 1, .tables = 1, .seed = 5 });
    defer cp.deinit(gpa);
    try std.testing.expectEqual(cp.key(&x, 0) ^ 1, cp.key(&nx, 0));
}

test "probes start with the own key and are distinct" {
    const gpa = std.testing.allocator;
    const x = [_]f32{ 0.3, -1.2, 0.8, 0.05, -0.4, 0.9, 0.2, -0.7, 1.1 };
    inline for (.{ Family.hyperplane, Family.cross_polytope }) |f| {
        const h = try Hasher.init(gpa, 9, .{ .family = f, .k = 2, .tables = 2, .seed = 3 });
        defer h.deinit(gpa);
        var out: [6]u64 = undefined;
        const n = try h.probes(&x, 1, &out);
        try std.testing.expectEqual(@as(usize, if (f == .hyperplane) 3 else 6), n);
        try std.testing.expectEqual(h.key(&x, 1), out[0]);
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
