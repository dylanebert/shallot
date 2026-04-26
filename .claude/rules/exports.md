---
paths:
    - "packages/shallot/src/**/*.ts"
    - "packages/shallot/package.json"
---

# Exports

## Tiers

- **`@dylanebert/shallot`** — public API. Components, registries, registration functions, shape factories, types. Everything a game developer needs
- **`@dylanebert/shallot/extras`** — opt-in plugins (orbit, tween, raytracing, etc.)
- **`@dylanebert/shallot/editor`** — editor-enabling libraries (document model, session sync)
- **`@dylanebert/shallot/runtime`** — platform layer (`Runtime`, `now`, `requestFrame`, `readFile`, `readBinary`)
- **`@dylanebert/shallot/*/core`** — non-public extension API per module. For building custom pipelines, tooling, diagnostics:
  - `render/core` — WGSL constants, surface compilation, batching, cull, pass utilities, light packing, mesh internals
  - `physics/core` — `PhysicsGPU` type, WGSL body data layouts, LBVH type
  - `audio/core` — `AudioState` + `Audio` resource, voice/transport pure functions, instrument/pattern registries
  - `compute/core` — `requestGPU`, GPU memory tracking, timestamp queries, compile timing
  - `transforms/core` — WASM transform buffer pointers, sync, compute
  - `ecs/core` — field proxies, component/relation introspection, schemas, entity inspection
  - `raytracing/core` — BVH construction/traversal, shader compilation, WGSL structs/constants, triangle extraction

## Barrel rules

Each standard plugin barrel (`index.ts`) exports its public API. What goes in the barrel vs a subpath:

- **Barrel:** components, resources, registration functions, types that game developers use
- **`*/core` subpath:** WGSL constants, shader compilation, GPU buffer packing, internal mesh/surface queries — things pipeline/extension authors need
- **Neither (internal):** implementation details consumed only within the module (viewport upload, scene buffer creation)

Cross-plugin imports within `standard/` use relative paths. Extension consumers (including raster and raytracing) import from `./render/core`.

## Registry pattern

Shared heavyweight data (meshes, surfaces, instruments, patterns, fonts) uses `Registry<T>` from `engine/utils/registry`. Registration functions (`mesh()`, `surface()`) wrap the registry with validation and side effects. Lookups go through the registry directly (`meshRegistry.get()`, `surfaceRegistry.getByName()`). No wrapper lookup functions.

## Type discipline

- No type aliases that just rename primitives (`type Foo = string`). Use the primitive directly
- Interfaces earn their place by having >1 field or methods
- Don't export types that are only used internally within one file

## Naming

Exports should not commonly conflict, requiring `as` renames at import sites. If a name is too generic (e.g. `readBuffer`), it belongs in a subpath or stays internal — not in the main barrel.
