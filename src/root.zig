//! zig-lsh: angular locality-sensitive hashing for approximate nearest-neighbour search.
//!
//! - `rotate`: a seeded random orthogonal transform (random signs and Hadamard, three rounds).
//! - `hash`: hyperplane and cross-polytope hash families, with multi-probe.
//! - `index`: L hash tables and the candidate union of a query.

const std = @import("std");

pub const rotate = @import("rotate.zig");
pub const hash = @import("hash.zig");
pub const index = @import("index.zig");

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(rotate);
    std.testing.refAllDecls(hash);
    std.testing.refAllDecls(index);
    _ = @import("fuzz.zig");
}
