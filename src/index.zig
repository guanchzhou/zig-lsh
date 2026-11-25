//! L hash tables over a set of vectors. Each table is an array of (key, id) pairs sorted by key,
//! then id; a lookup is a binary search. `candidates` returns the sorted, de-duplicated union of
//! the buckets that a query's own keys and probe keys hit in every table.

const std = @import("std");
const hash = @import("hash.zig");

pub const Entry = struct { key: u64, id: u32 };

pub const Index = struct {
    params: hash.Params,
    dims: usize,
    n: usize,
    tables: []const []const Entry,

    pub fn deinit(self: Index, gpa: std.mem.Allocator) void {
        for (self.tables) |t| gpa.free(t);
        gpa.free(self.tables);
    }

    /// Entries of `table` whose key is `key`.
    pub fn bucket(self: Index, table: u32, key: u64) []const Entry {
        if (table >= self.tables.len) return &.{};
        const t = self.tables[table];
        const lo = std.sort.lowerBound(Entry, t, key, cmpKey);
        if (lo == t.len or t[lo].key != key) return &.{};
        const tail = t[lo..];
        const probe_limit = @min(tail.len, 16);
        var hi: usize = 1;
        while (hi < probe_limit and tail[hi].key == key) : (hi += 1) {}
        if (hi == probe_limit and hi < tail.len and tail[hi].key == key) {
            hi += std.sort.upperBound(Entry, tail[hi..], key, cmpKey);
        }
        return tail[0..hi];
    }
};

fn cmpKey(key: u64, e: Entry) std.math.Order {
    return std.math.order(key, e.key);
}

fn entryLess(_: void, a: Entry, b: Entry) bool {
    return a.key < b.key or (a.key == b.key and a.id < b.id);
}

pub const Error = error{ RaggedData, TooManyVectors, InvalidIndex } || hash.Error || std.mem.Allocator.Error;

/// Hashes the `data.len / dims` row-major vectors of `data` into `params.tables` tables.
pub fn build(gpa: std.mem.Allocator, data: []const f32, dims: usize, params: hash.Params) Error!Index {
    try hash.validate(params, dims);
    if (data.len % dims != 0) return error.RaggedData;
    for (data) |v| if (!std.math.isFinite(v)) return error.NonFiniteValue;
    const n = data.len / dims;
    if (n > std.math.maxInt(u32)) return error.TooManyVectors;
    var h = try hash.Hasher.init(gpa, dims, params);
    defer h.deinit(gpa);
    const tables = try gpa.alloc([]const Entry, params.tables);
    var built: usize = 0;
    errdefer {
        for (tables[0..built]) |t| gpa.free(t);
        gpa.free(tables);
    }
    for (0..params.tables) |t| {
        const entries = try gpa.alloc(Entry, n);
        errdefer gpa.free(entries);
        for (entries, 0..) |*e, i| e.* = .{ .key = try h.key(data[i * dims ..][0..dims], @intCast(t)), .id = @intCast(i) };
        std.mem.sort(Entry, entries, {}, entryLess);
        tables[t] = entries;
        built += 1;
    }
    return .{ .params = params, .dims = dims, .n = n, .tables = tables };
}

/// Ids of every vector sharing a bucket with `x` in any table, probing `probes` extra keys per
/// table (0 = own key only). Sorted ascending, without duplicates. The caller frees the slice.
pub fn candidates(gpa: std.mem.Allocator, ix: Index, x: []const f32, probes: usize) Error![]u32 {
    if (probes >= hash.max_probes) return error.TooManyProbes;
    if (x.len != ix.dims) return error.DimensionMismatch;
    if (ix.tables.len != ix.params.tables) return error.InvalidIndex;
    for (x) |v| if (!std.math.isFinite(v)) return error.NonFiniteValue;
    var h = try hash.Hasher.init(gpa, ix.dims, ix.params);
    defer h.deinit(gpa);
    var seen = try std.DynamicBitSetUnmanaged.initEmpty(gpa, ix.n);
    defer seen.deinit(gpa);
    var keys: [hash.max_probes]u64 = undefined;
    for (0..ix.params.tables) |t| {
        const got = try h.probes(x, @intCast(t), keys[0 .. probes + 1]);
        for (keys[0..got]) |k| {
            for (ix.bucket(@intCast(t), k)) |e| {
                if (e.id >= ix.n) return error.InvalidIndex;
                seen.set(e.id);
            }
        }
    }
    const out = try gpa.alloc(u32, seen.count());
    var it = seen.iterator(.{});
    var i: usize = 0;
    while (it.next()) |id| : (i += 1) out[i] = @intCast(id);
    return out;
}

test "a vector finds itself in every table" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(9);
    const r = prng.random();
    var data: [40 * 16]f32 = undefined;
    for (&data) |*v| v.* = r.floatNorm(f32);
    inline for (.{ hash.Family.hyperplane, hash.Family.cross_polytope }) |f| {
        const ix = try build(gpa, &data, 16, .{ .family = f, .k = if (f == .hyperplane) 6 else 1, .tables = 4, .seed = 1 });
        defer ix.deinit(gpa);
        for (0..40) |i| {
            const c = try candidates(gpa, ix, data[i * 16 ..][0..16], 0);
            defer gpa.free(c);
            try std.testing.expect(std.mem.indexOfScalar(u32, c, @intCast(i)) != null);
        }
    }
}

test "more probes never lose candidates" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(4);
    const r = prng.random();
    var data: [60 * 12]f32 = undefined;
    for (&data) |*v| v.* = r.floatNorm(f32);
    const ix = try build(gpa, &data, 12, .{ .family = .cross_polytope, .k = 2, .tables = 3, .seed = 2 });
    defer ix.deinit(gpa);
    const q = data[0..12];
    const a = try candidates(gpa, ix, q, 0);
    defer gpa.free(a);
    const b = try candidates(gpa, ix, q, 4);
    defer gpa.free(b);
    for (a) |id| try std.testing.expect(std.mem.indexOfScalar(u32, b, id) != null);
    try std.testing.expectError(error.TooManyProbes, candidates(gpa, ix, q, hash.max_probes));
}

test "candidates rejects a query of the wrong width" {
    const gpa = std.testing.allocator;
    const data = [_]f32{ 1, 0, 0, 1, -1, 0 };
    const ix = try build(gpa, &data, 2, .{ .family = .hyperplane, .k = 2, .tables = 2, .seed = 1 });
    defer ix.deinit(gpa);
    try std.testing.expectError(error.DimensionMismatch, candidates(gpa, ix, &.{1}, 0));
    try std.testing.expectError(error.DimensionMismatch, candidates(gpa, ix, &.{ 1, 2, 3 }, 1));
}

fn buildAlloc(gpa: std.mem.Allocator, family: hash.Family) !void {
    const data = [_]f32{ 1, 0, 0.5, 0, 1, -0.5, -1, 0, 0.2, 0.3, 0.3, 0.3 };
    const ix = try build(gpa, &data, 3, .{ .family = family, .k = 1, .tables = 3, .seed = 4 });
    defer ix.deinit(gpa);
    const c = try candidates(gpa, ix, data[0..3], 2);
    gpa.free(c);
}

test "build and candidates free everything on allocation failure" {
    inline for (.{ hash.Family.hyperplane, hash.Family.cross_polytope }) |f| {
        try std.testing.checkAllAllocationFailures(std.testing.allocator, buildAlloc, .{f});
    }
}

test "build and candidates reject non-finite vectors" {
    const gpa = std.testing.allocator;
    const params: hash.Params = .{ .family = .hyperplane, .k = 2, .tables = 2, .seed = 1 };
    try std.testing.expectError(error.NonFiniteValue, build(gpa, &.{ 1, std.math.nan(f32), 0, 1 }, 2, params));
    try std.testing.expectError(error.NonFiniteValue, build(gpa, &.{ 1, std.math.inf(f32), 0, 1 }, 2, params));
    const ix = try build(gpa, &.{ 1, 0, 0, 1 }, 2, params);
    defer ix.deinit(gpa);
    try std.testing.expectError(error.NonFiniteValue, candidates(gpa, ix, &.{ std.math.nan(f32), 1 }, 0));
    try std.testing.expectError(error.NonFiniteValue, candidates(gpa, ix, &.{ -std.math.inf(f32), 1 }, 1));
}

test "bucket bounds and linear/binary lookup sizes" {
    var entries: [50]Entry = undefined;
    for (0..3) |i| entries[i] = .{ .key = 10, .id = @intCast(i) };
    for (3..19) |i| entries[i] = .{ .key = 20, .id = @intCast(i) };
    for (19..50) |i| entries[i] = .{ .key = 30, .id = @intCast(i) };
    const tables = [_][]const Entry{&entries};
    const ix: Index = .{
        .params = .{ .family = .hyperplane, .k = 1, .tables = 1, .seed = 0 },
        .dims = 2,
        .n = 50,
        .tables = &tables,
    };
    try std.testing.expectEqual(@as(usize, 0), ix.bucket(1, 10).len);
    try std.testing.expectEqual(@as(usize, 0), ix.bucket(std.math.maxInt(u32), 10).len);
    try std.testing.expectEqual(@as(usize, 0), ix.bucket(0, 15).len);
    try std.testing.expectEqual(@as(usize, 3), ix.bucket(0, 10).len);
    try std.testing.expectEqual(@as(usize, 16), ix.bucket(0, 20).len);
    try std.testing.expectEqual(@as(usize, 31), ix.bucket(0, 30).len);
}

test "candidates rejects invalid table count and out-of-range entry ids" {
    const gpa = std.testing.allocator;
    const data = [_]f32{ 1, 0, 0, 1 };
    const ix = try build(gpa, &data, 2, .{ .family = .hyperplane, .k = 2, .tables = 2, .seed = 1 });
    defer ix.deinit(gpa);

    var bad_tables = ix;
    bad_tables.tables = ix.tables[0..1];
    try std.testing.expectError(error.InvalidIndex, candidates(gpa, bad_tables, &.{ 1, 0 }, 0));

    const bad_entries = [_]Entry{.{ .key = ix.tables[0][0].key, .id = 99 }};
    const corrupt_tables = [_][]const Entry{ &bad_entries, ix.tables[1] };
    var bad_id = ix;
    bad_id.tables = &corrupt_tables;
    const hit_vec = data[@as(usize, ix.tables[0][0].id) * 2 ..][0..2];
    try std.testing.expectError(error.InvalidIndex, candidates(gpa, bad_id, hit_vec, 0));
}
