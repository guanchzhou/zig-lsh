const std = @import("std");
const lsh = @import("lsh");
const build_options = @import("build_options");
const hash = lsh.hash;
const index = lsh.index;

const usage =
    \\usage: zig-lsh COMMAND < input.json
    \\
    \\  keys    {"data":[[..]..],"family":F,"k":K,"tables":L,"seed":"HEX"?,"probes":P?}
    \\          -> one line per row: {"keys":["HEX"..]} with one key per table, and with P > 0
    \\             also "probes":[["HEX"..]..], each table's own key followed by up to P probe keys
    \\  search  {"data":[[..]..],"queries":[[..]..],"family":F,"k":K,"tables":L,"seed":"HEX"?,
    \\           "probes":P?,"top":T?}
    \\          -> {"results":[{"candidates":C,"top":[i..]}..]}: the candidate count of each query
    \\             and its T best candidates by exact cosine
    \\  version
    \\
    \\  F is "hyperplane" (K bits per table) or "cross-polytope" (K hashes per table).
    \\
;

const default_seed: u64 = 0x9e3779b97f4a7c15;

fn fail(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("zig-lsh: " ++ fmt ++ "\n", args);
    std.process.exit(1);
}

fn parse(comptime T: type, arena: std.mem.Allocator, text: []const u8) T {
    return std.json.parseFromSliceLeaky(T, arena, text, .{ .ignore_unknown_fields = true }) catch |e|
        fail("invalid input: {s}", .{@errorName(e)});
}

fn flatten(arena: std.mem.Allocator, rows: []const []const f32, what: []const u8) !struct { data: []f32, d: usize } {
    if (rows.len == 0) fail("{s} needs at least one row", .{what});
    const d = rows[0].len;
    if (d == 0) fail("{s} rows must not be empty", .{what});
    const data = try arena.alloc(f32, rows.len * d);
    for (rows, 0..) |r, i| {
        if (r.len != d) fail("{s} row {d} has {d} values, expected {d}", .{ what, i, r.len, d });
        @memcpy(data[i * d ..][0..d], r);
    }
    return .{ .data = data, .d = d };
}

fn familyOf(text: []const u8) hash.Family {
    if (std.mem.eql(u8, text, "hyperplane")) return .hyperplane;
    if (std.mem.eql(u8, text, "cross-polytope")) return .cross_polytope;
    fail("family must be hyperplane or cross-polytope: {s}", .{text});
}

fn seedOf(text: ?[]const u8) u64 {
    const s = text orelse return default_seed;
    return std.fmt.parseInt(u64, s, 16) catch fail("seed must be hexadecimal: {s}", .{s});
}

fn writeHex(out: *std.Io.Writer, k: u64) !void {
    try out.print("\"{x:0>16}\"", .{k});
}

fn cosine(a: []const f32, b: []const f32) f64 {
    var ab: f64 = 0;
    var aa: f64 = 0;
    var bb: f64 = 0;
    for (a, b) |x, y| {
        ab += @as(f64, x) * y;
        aa += @as(f64, x) * x;
        bb += @as(f64, y) * y;
    }
    return if (aa == 0 or bb == 0) 0 else ab / @sqrt(aa * bb);
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    var buf: [64 * 1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &buf);
    const out = &stdout.interface;
    defer out.flush() catch {};
    if (args.len != 2) {
        std.debug.print("{s}", .{usage});
        std.process.exit(2);
    }
    const cmd: []const u8 = args[1];
    if (std.mem.eql(u8, cmd, "version") or std.mem.eql(u8, cmd, "--version")) {
        try out.print("zig-lsh {s}\n", .{build_options.version});
        return;
    }
    if (std.mem.eql(u8, cmd, "help") or std.mem.eql(u8, cmd, "--help") or std.mem.eql(u8, cmd, "-h")) {
        try out.writeAll(usage);
        return;
    }
    if (!std.mem.eql(u8, cmd, "keys") and !std.mem.eql(u8, cmd, "search")) {
        std.debug.print("zig-lsh: unknown command: {s}\n{s}", .{ cmd, usage });
        std.process.exit(2);
    }
    var in_buf: [64 * 1024]u8 = undefined;
    var stdin = std.Io.File.stdin().readerStreaming(init.io, &in_buf);
    const text = stdin.interface.allocRemaining(arena, .limited(256 * 1024 * 1024)) catch |e| switch (e) {
        error.ReadFailed => return stdin.err.?,
        else => |x| return x,
    };

    const In = struct {
        data: []const []const f32,
        queries: []const []const f32 = &.{},
        family: []const u8 = "hyperplane",
        k: u8 = 8,
        tables: u32 = 1,
        seed: ?[]const u8 = null,
        probes: usize = 0,
        top: usize = 10,
    };
    const in = parse(In, arena, text);
    const m = try flatten(arena, in.data, "data");
    const params: hash.Params = .{ .family = familyOf(in.family), .k = in.k, .tables = in.tables, .seed = seedOf(in.seed) };
    hash.validate(params, m.d) catch |e| fail("{s}", .{@errorName(e)});
    if (in.probes + 1 > hash.max_probes) fail("probes must be at most {d}", .{hash.max_probes - 1});

    if (std.mem.eql(u8, cmd, "keys")) {
        var h = try hash.Hasher.init(arena, m.d, params);
        var keys: [hash.max_probes]u64 = undefined;
        for (0..in.data.len) |i| {
            const x = m.data[i * m.d ..][0..m.d];
            try out.writeAll("{\"keys\":[");
            for (0..params.tables) |t| {
                if (t > 0) try out.writeAll(",");
                try writeHex(out, try h.key(x, @intCast(t)));
            }
            try out.writeAll("]");
            if (in.probes > 0) {
                try out.writeAll(",\"probes\":[");
                for (0..params.tables) |t| {
                    if (t > 0) try out.writeAll(",");
                    const got = try h.probes(x, @intCast(t), keys[0 .. in.probes + 1]);
                    try out.writeAll("[");
                    for (keys[0..got], 0..) |k, j| {
                        if (j > 0) try out.writeAll(",");
                        try writeHex(out, k);
                    }
                    try out.writeAll("]");
                }
                try out.writeAll("]");
            }
            try out.writeAll("}\n");
        }
        return;
    }

    const qs = try flatten(arena, in.queries, "queries");
    if (qs.d != m.d) fail("queries have {d} values, data has {d}", .{ qs.d, m.d });
    if (in.top == 0) fail("top must be at least 1", .{});
    const ix = index.build(arena, m.data, m.d, params) catch |e| fail("{s}", .{@errorName(e)});
    try out.writeAll("{\"results\":[");
    for (0..in.queries.len) |qi| {
        const q = qs.data[qi * m.d ..][0..m.d];
        const cand = try index.candidates(arena, ix, q, in.probes);
        const Item = struct { i: u32, s: f64 };
        const items = try arena.alloc(Item, cand.len);
        for (items, cand) |*it, i| it.* = .{ .i = i, .s = cosine(m.data[@as(usize, i) * m.d ..][0..m.d], q) };
        std.mem.sort(Item, items, {}, struct {
            fn f(_: void, a: Item, b: Item) bool {
                return a.s > b.s or (a.s == b.s and a.i < b.i);
            }
        }.f);
        if (qi > 0) try out.writeAll(",");
        try out.print("{{\"candidates\":{d},\"top\":[", .{cand.len});
        for (items[0..@min(in.top, items.len)], 0..) |it, j| {
            if (j > 0) try out.writeAll(",");
            try out.print("{d}", .{it.i});
        }
        try out.writeAll("]}");
    }
    try out.writeAll("]}\n");
}
