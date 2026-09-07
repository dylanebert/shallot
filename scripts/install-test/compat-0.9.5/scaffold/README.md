# compat-app

A shallot project.

## Develop

```bash
bun install
bunx shallot dev
```

`bun install` fetches the engine. `bunx shallot dev` runs the project with hot
reload. Edit `src/spin.ts` (a plugin) and `shallot.json` (the manifest: scene +
plugin enablement) in your IDE.

## Ship

```bash
bunx shallot build
```

Builds a web bundle to `dist/`. Native targets
(`--target windows|mac|linux`) download a prebuilt release shell when one exists
for your installed version, and otherwise fall back to compiling from source,
which needs the Rust toolchain and target system dependencies; see the repo
README for the per-target table.
