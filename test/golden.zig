//! Replays test/golden.json, the outputs of the independent reference in test/reference.py.
//! Keys, probe keys and candidate sets must match exactly.

const std = @import("std");
const lsh = @import("lsh");
const hash = lsh.hash;
const index = lsh.index;

const Golden = struct {
    cases: []const struct {
        d: usize,
        family: []const u8,
        k: u8,
        tables: u32,
        probes: usize,
        seed: []const u8,
        data: []const []const f32,
        queries: []const []const f32,
        keys: []const []const []const u8,
        query_probes: []const []const []const []const u8,
        candidates: []const []const u32,
    },
};

fn hex(s: []const u8) !u64 {
    return std.fmt.parseInt(u64, s, 16);
}

test "keys, probes and candidates match the NumPy reference" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const g = try std.json.parseFromSliceLeaky(Golden, arena, @embedFile("golden.json"), .{});
    try std.testing.expect(g.cases.len >= 50);
    for (g.cases) |c| {
        const family: hash.Family = if (std.mem.eql(u8, c.family, "hyperplane")) .hyperplane else .cross_polytope;
        const params: hash.Params = .{ .family = family, .k = c.k, .tables = c.tables, .seed = try hex(c.seed) };
        const h = try hash.Hasher.init(arena, c.d, params);
        const data = try arena.alloc(f32, c.data.len * c.d);
        for (c.data, 0..) |row, i| @memcpy(data[i * c.d ..][0..c.d], row);
        for (c.keys, 0..) |row, i| {
            for (row, 0..) |want, t| try std.testing.expectEqual(try hex(want), h.key(data[i * c.d ..][0..c.d], @intCast(t)));
        }
        const ix = try index.build(arena, data, c.d, params);
        var out: [hash.max_probes]u64 = undefined;
        for (c.queries, c.query_probes, c.candidates) |q, qp, want_cand| {
            for (qp, 0..) |want, t| {
                const got = try h.probes(q, @intCast(t), out[0 .. c.probes + 1]);
                try std.testing.expectEqual(want.len, got);
                for (want, out[0..got]) |w, o| try std.testing.expectEqual(try hex(w), o);
            }
            const cand = try index.candidates(arena, ix, q, c.probes);
            try std.testing.expectEqualSlices(u32, want_cand, cand);
        }
    }
}
