# physics fixtures — frozen historical regression evidence

The 53 `*.json` files here are frozen historical regression evidence: per-step FNV-1a world-state hashes
plus periodic body-state dumps. `src/transitional/physics/solver/step.fixture.ts` rebuilds each scene through
the engine's public API, steps it, and asserts its hash equals the retained value at every step — at
single-thread and every thread count. The values are intentionally preserved as reviewed history; they are
not an authority or a minting input for new expected values.

These files are committed, not gitignored, so the historical regression contract remains self-contained and
runnable without a C toolchain or sibling checkout. They sit under `tests/` (outside `files: ["src"]`), so
they never ship in the npm package.

## Provenance

- **Historical source:** the committed values came from an earlier Box3D compatibility boundary and remain
  immutable regression evidence.
- **Current authority:** target updates, migration, generation, and reproduction belong to the standalone
  `projects/box3d-oracle` workspace, which uses official Box3D and records each generated bundle's provenance.
- **Current engine path:** the solver uses canonical graph coloring, with the real capacity overflow color
  retained for constraints that cannot fit. These fixtures do not authorize a separate solver configuration.

## Updating oracle evidence

Keep these historical files frozen. For a deliberate official-upstream sync, use the commands and schemas in
`projects/box3d-oracle`; do not add a generator or pin under the Shallot repository. Never hand-edit a fixture
to make a test pass: a mismatch is evidence to investigate, not a value to adjust.
