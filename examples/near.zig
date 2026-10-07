const std = @import("std");
const lsh = @import("lsh");
const index = lsh.index;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var prng = std.Random.DefaultPrng.init(42);
    const r = prng.random();
    const n = 2000;
    const d = 64;
    const data = try gpa.alloc(f32, n * d);
    defer gpa.free(data);
    for (data) |*x| x.* = r.floatNorm(f32);

    // Cross-polytope LSH: one hash of 128 buckets per table, eight tables.
    const ix = try index.build(gpa, data, d, .{ .family = .cross_polytope, .k = 1, .tables = 8, .seed = 0x9e3779b97f4a7c15 });
    defer ix.deinit(gpa);

    // A query close to vector 123 shares a bucket with it in some table, and the candidate set is a
    // small fraction of the data.
    var query: [d]f32 = undefined;
    for (&query, data[123 * d ..][0..d]) |*q, x| q.* = x + 0.2 * r.floatNorm(f32);
    const cand = try index.candidates(gpa, ix, &query, 2);
    defer gpa.free(cand);
    std.debug.assert(std.mem.indexOfScalar(u32, cand, 123) != null);
    std.debug.assert(cand.len < n / 4);
}
