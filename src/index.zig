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
        const t = self.tables[table];
        const lo = std.sort.lowerBound(Entry, t, key, cmpKey);
        const hi = std.sort.upperBound(Entry, t, key, cmpKey);
        return t[lo..hi];
    }
};

fn cmpKey(key: u64, e: Entry) std.math.Order {
    return std.math.order(key, e.key);
}

fn entryLess(_: void, a: Entry, b: Entry) bool {
    return a.key < b.key or (a.key == b.key and a.id < b.id);
}

pub const Error = error{ RaggedData, TooManyVectors } || hash.Error || std.mem.Allocator.Error;

/// Hashes the `data.len / dims` row-major vectors of `data` into `params.tables` tables.
pub fn build(gpa: std.mem.Allocator, data: []const f32, dims: usize, params: hash.Params) Error!Index {
    try hash.validate(params, dims);
    if (data.len % dims != 0) return error.RaggedData;
    const n = data.len / dims;
    if (n > std.math.maxInt(u32)) return error.TooManyVectors;
    const h = try hash.Hasher.init(gpa, dims, params);
    defer h.deinit(gpa);
    const tables = try gpa.alloc([]const Entry, params.tables);
    var built: usize = 0;
    errdefer {
        for (tables[0..built]) |t| gpa.free(t);
        gpa.free(tables);
    }
    for (0..params.tables) |t| {
        const entries = try gpa.alloc(Entry, n);
        for (entries, 0..) |*e, i| e.* = .{ .key = h.key(data[i * dims ..][0..dims], @intCast(t)), .id = @intCast(i) };
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
    const h = try hash.Hasher.init(gpa, ix.dims, ix.params);
    defer h.deinit(gpa);
    var seen = try std.DynamicBitSetUnmanaged.initEmpty(gpa, ix.n);
    defer seen.deinit(gpa);
    var keys: [hash.max_probes]u64 = undefined;
    for (0..ix.params.tables) |t| {
        const got = try h.probes(x, @intCast(t), keys[0 .. probes + 1]);
        for (keys[0..got]) |k| {
            for (ix.bucket(@intCast(t), k)) |e| seen.set(e.id);
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
