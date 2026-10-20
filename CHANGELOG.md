# Changelog

## 0.1.1 - 2026-10-20

- Return `error.TooManyProbes` in `Index.candidates` when `probes >= hash.max_probes` instead of slicing past the stack buffer.
- Hoist base cross-polytope rotations outside the multi-probe candidate loop in `Hasher.probes` and vectorise `rotate.fwht` with `@Vector(4, f64)`.
- Add GitHub Actions CI workflow.

## 0.1.0 - 2026-10-07

- `lsh.rotate`: seeded fast Hadamard rotation (three rounds of random signs).
- `lsh.hash`: hyperplane and cross-polytope hash families with multi-probe; keys depend only on direction.
- `lsh.index`: L tables as sorted (key, id) arrays; candidate union of a query's keys and probes.
- `zig-lsh keys` and `zig-lsh search` on the command line.
- Golden tests against an independent NumPy reference (96 cases), fuzz invariants, an example.
