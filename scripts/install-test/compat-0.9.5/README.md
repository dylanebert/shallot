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

Nothing here is regenerated. Replacing it means publishing a new release and freezing that one.
