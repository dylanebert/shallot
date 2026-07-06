# starter

A shallot project.

## Develop

```bash
bun install
bunx shallot dev
```

`bun install` fetches the engine. `bunx shallot dev` runs the project with hot
reload; `bunx shallot` opens it in the editor. Edit `src/spin.ts` (a plugin) and
`shallot.json` (the manifest: scene + plugin enablement) in your IDE.

## Ship

```bash
bunx shallot build
```

Builds a web bundle to `dist/`. For native targets:
`bunx shallot build --target windows|mac|linux` (add `--release` for an optimized build).
