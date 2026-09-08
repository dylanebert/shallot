# Compatibility baseline — 0.9.5

The last actually-published release, frozen before the package-boundary moves so a candidate can be
compared against a real previous consumer rather than against today's templates.

- `PIN.json` — both tarballs' versions, registry integrity and shasum, file counts and unpacked sizes.
  `bun run scripts/check-compat-pin.ts` re-reads them; a fetched tarball whose digest disagrees is a
  different artifact, not a newer one.
- `scaffold/` — the project `bunx create-shallot@0.9.5` emits, byte-for-byte. Excluded from Biome, tsc
  and the scene formatter: reformatting it would destroy the thing it is here to preserve.
- `engine-package.json` / `engine-files.txt` — the published engine's manifest and its 424-file tarball
  inventory, so an export, a `files` entry or a packed artifact that disappears is visible as a diff.

Run `bun run scripts/check-compat-pin.ts --fetch` to realize both archives into ignored `tarballs/`.
It writes only after both archives match the pinned SHA-512 and SHA-1 digests. Plain `bun run check`
requires those bytes present and matching, never downloads or writes them.

`PIN.json`'s `inputs` map names original engine archive paths by role, checked against the inventory:
ejected Vite/TGSL documentation and config, the GPU-particles plugin's source/manifests/scene, and
CLI/default implementation. The retained archives supply those bytes for compatibility extraction;
`scaffold/` remains the original default emitted project. No old input is regenerated from new templates.

Replacing this baseline means publishing a new release and freezing that one.
