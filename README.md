# zig-lsh

Angular locality-sensitive hashing in Zig, for approximate nearest-neighbour search over embedding
vectors:

- **Cross-polytope LSH** (Andoni, Indyk, Laarhoven, Razenshteyn and Schmidt, 2015). Each hash is the index and sign of
  the largest-magnitude coordinate after a random rotation, one of 2n buckets.
- **Hyperplane LSH** (Charikar, 2002). Each bit is the sign of one rotated coordinate.
- **L tables and multi-probe.** Hyperplane probes flip the least certain bit first; cross-polytope probes substitute
  the runner-up coordinate with the smallest margin first.

Rotations are a seeded fast Hadamard transform with three rounds of random signs, which costs
O(n log n) per vector. The library is pure Zig 0.17 with no dependencies, released under the MIT
licence. It is checked against an independent NumPy reference that builds every rotation as a dense
matrix.

## Install

```sh
zig fetch --save git+https://github.com/guanchzhou/zig-lsh
```

```zig
const lsh = b.dependency("zig_lsh", .{}).module("lsh");
exe.root_module.addImport("lsh", lsh);
```

## Library

```zig
const lsh = @import("lsh");

// 16 tables, one cross-polytope hash each.
const ix = try lsh.index.build(gpa, data, dims, .{ .family = .cross_polytope, .k = 1, .tables = 16, .seed = 42 });
defer ix.deinit(gpa);

// Ids sharing a bucket with the query in any table, with two extra probes per table.
const cand = try lsh.index.candidates(gpa, ix, query, 2);
defer gpa.free(cand);
```

`lsh.hash.Hasher` gives the keys (`key`) and probe keys (`probes`) of a vector directly, for callers
that keep their own tables. Both families depend only on a vector's direction, so any positive scale
gives the same keys.

- **`k` for cross-polytope:** the number of hashes per table, with (2n)^k at most 2^64 for padded length n.
- **`k` for hyperplane:** the number of bits per table, at most 64.

## Command line

```sh
zig build -Doptimize=ReleaseFast
echo '{"data":[[0.3,-1.2,0.8,0.5],[1.1,0.2,-0.4,0.9]],"family":"cross-polytope","k":1,"tables":2}' | zig-out/bin/zig-lsh keys
# {"keys":["0000000000000001","0000000000000005"]}
# {"keys":["0000000000000002","0000000000000006"]}
```

- `keys` writes one line per input row, with the key of each table. With `"probes":P` it also writes
  each table's probe keys.
- `search` builds the tables over `data` and returns, for each query, the candidate count and the best
  candidates by exact cosine.

`zig-lsh --help` lists every field.

## Tests

```sh
zig build test                            # unit, golden, fuzz-corpus, example and CLI tests
uv run --with numpy python test/reference.py   # regenerate test/golden.json
zig build test --fuzz                     # explore further
```

`test/golden.json` holds 96 cases (both families, dimensions 3 to 64, one to three tables, up to
three probes). Keys, probe keys and candidate sets must match exactly. Two kinds of mutation make the
golden test fail: inverting the cross-polytope sign bit, and shifting the rotation-seed formula.

When an input is padded to a power of two, a rotation row can be exactly zero on the input's
coordinates, and two rotated coordinates can have exactly equal magnitudes. Rounding then decides
signs and ties. To keep the result the same on every platform, magnitudes are compared on a grid of
1e-9 times the vector's norm, and a hyperplane bit is set only above 1e-12 times the norm.

## Licence

MIT. See `LICENSE`.
