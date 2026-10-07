//! Command-line cases run by `zig build test-cli`. `stdout` is compared
//! exactly, `stderr` as a substring.

pub const Case = struct {
    name: []const u8,
    args: []const []const u8,
    stdin: ?[]const u8 = null,
    stdout: ?[]const u8 = null,
    stderr: ?[]const u8 = null,
    exit: u8 = 0,
};

pub const cases = [_]Case{
    .{ .name = "error: family", .args = &.{"keys"}, .stdin = "{\"data\":[[1,0]],\"family\":\"spherical\"}", .stderr = "family must be hyperplane or cross-polytope: spherical", .exit = 1 },
    .{ .name = "error: k out of range", .args = &.{"keys"}, .stdin = "{\"data\":[[1,0]],\"family\":\"hyperplane\",\"k\":3}", .stderr = "KOutOfRange", .exit = 1 },
    .{ .name = "error: no tables", .args = &.{"keys"}, .stdin = "{\"data\":[[1,0]],\"tables\":0,\"k\":1}", .stderr = "NoTables", .exit = 1 },
    .{ .name = "error: ragged rows", .args = &.{"keys"}, .stdin = "{\"data\":[[1,2],[3]]}", .stderr = "data row 1 has 1 values, expected 2", .exit = 1 },
    .{ .name = "error: query width", .args = &.{"search"}, .stdin = "{\"data\":[[1,2]],\"queries\":[[1]],\"k\":1}", .stderr = "queries have 1 values, data has 2", .exit = 1 },
    .{ .name = "error: too many probes", .args = &.{"keys"}, .stdin = "{\"data\":[[1,2]],\"k\":1,\"probes\":65}", .stderr = "probes must be at most 64", .exit = 1 },
    .{ .name = "error: truncated JSON", .args = &.{"keys"}, .stdin = "{", .stderr = "invalid input", .exit = 1 },
    .{ .name = "error: unknown command", .args = &.{"bogus"}, .stderr = "unknown command: bogus", .exit = 2 },
    .{ .name = "error: no command prints usage", .args = &.{}, .stderr = "usage: zig-lsh COMMAND", .exit = 2 },
    .{ .name = "help: exits 0", .args = &.{"--help"} },
};
