# Changelog

## 0.1.3 - 2026-11-25

- Reject non-finite inputs (`error.NonFiniteValue`) and out-of-range table indices (`error.TableOutOfRange`) in `Hasher` and `index`, and make `altLess` total-ordered.
- Validate `Index` invariants in `candidates` and guard `Index.bucket`.
- Fix CLI `probes` overflow, validate ragged rows before allocation, and fix 32-bit shift portability.
- Use single-pass top-(want+1) probe selection with precomputed `inv_unit` and narrowed `Index.bucket` lookup.

## 0.1.2 - 2026-10-25

- `Hasher.key` and `Hasher.probes` take `*Hasher`, since they write into the hasher's buffer, and return errors. Callers declare the hasher with `var` and use `try`. New `Hasher.keyWith`, `Hasher.probesWith` and `Hasher.scratchLen` only read the hasher and use a caller-supplied scratch buffer, so one hasher can serve several threads.
- `Hasher.key`, `Hasher.probes` and `index.candidates` return `error.DimensionMismatch` when the vector's width is not the hasher's or index's `dims`, instead of asserting.
- `Hasher.probes` sizes its stack buffer to the most alternatives a valid table can produce (567, about 14 KB) instead of 4,160 (about 133 KB).
- `rotate.paddedLen` returns `error.DimensionTooLarge` instead of reaching `unreachable`, and `rotate.apply` returns `error.DimensionMismatch` instead of asserting; `hash.validate` reports both.
- Cap CLI `stdin` reads at 256 MiB.

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
