//! Fuzz targets checking invariants that must hold for every input. `zig build test` runs every
//! input in `src/fuzz-corpus/`; `zig build test --fuzz` explores further.

const std = @import("std");
const hash = @import("hash.zig");
const index = @import("index.zig");
const Smith = std.testing.Smith;

const corpus = [_][]const u8{
    @embedFile("fuzz-corpus/00"),
    @embedFile("fuzz-corpus/01"),
    @embedFile("fuzz-corpus/02"),
    @embedFile("fuzz-corpus/03"),
};

fn upTo(smith: *Smith, lo: usize, hi: usize) usize {
    return smith.valueRangeAtMost(u32, @intCast(lo), @intCast(hi));
}

fn invariants(_: void, smith: *Smith) anyerror!void {
    const gpa = std.testing.allocator;
    const n = upTo(smith, 1, 20);
    const d = upTo(smith, 1, 33);
    var data: [20 * 33]f32 = undefined;
    for (data[0 .. n * d]) |*x| x.* = @as(f32, @floatFromInt(smith.value(i16))) / 1024;
    const family: hash.Family = if (smith.value(bool)) .hyperplane else .cross_polytope;
    const padded = @import("rotate.zig").paddedLen(d);
    const k: u8 = switch (family) {
        .hyperplane => @intCast(upTo(smith, 1, @min(64, padded))),
        .cross_polytope => @intCast(upTo(smith, 1, @as(usize, 64) / @as(usize, std.math.log2_int(usize, 2 * padded)))),
    };
    const params: hash.Params = .{ .family = family, .k = k, .tables = @intCast(upTo(smith, 1, 3)), .seed = smith.value(u64) };
    const ix = try index.build(gpa, data[0 .. n * d], d, params);
    defer ix.deinit(gpa);
    const h = try hash.Hasher.init(gpa, d, params);
    defer h.deinit(gpa);
    const probes = upTo(smith, 0, 6);
    var doubled: [33]f32 = undefined;
    for (0..n) |i| {
        const x = data[i * d ..][0..d];
        for (doubled[0..d], x) |*a, b| a.* = 2 * b;
        var keys: [hash.max_probes]u64 = undefined;
        for (0..params.tables) |t| {
            const key = h.key(x, @intCast(t));
            try std.testing.expectEqual(key, h.key(x, @intCast(t)));
            try std.testing.expectEqual(key, h.key(doubled[0..d], @intCast(t)));
            const got = try h.probes(x, @intCast(t), keys[0 .. probes + 1]);
            try std.testing.expect(got >= 1 and got <= probes + 1);
            try std.testing.expectEqual(key, keys[0]);
            for (keys[0..got], 0..) |a, j| for (keys[j + 1 .. got]) |b| try std.testing.expect(a != b);
        }
        const c = try index.candidates(gpa, ix, x, probes);
        defer gpa.free(c);
        try std.testing.expect(std.mem.indexOfScalar(u32, c, @intCast(i)) != null);
        for (c[1..], c[0 .. c.len - 1]) |b, a| try std.testing.expect(a < b);
    }
}

test "fuzz: hashing and index invariants" {
    try std.testing.fuzz({}, invariants, .{ .corpus = &corpus });
}
