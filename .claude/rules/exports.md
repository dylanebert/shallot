---
paths:
    - "packages/shallot/src/**/*.ts"
    - "packages/shallot/package.json"
    - "packages/shallot-tooling/src/**/*.ts"
    - "packages/shallot-tooling/package.json"
    - "examples/showcase/ocean/src/ocean/**"
---

# Exports

## Tiers

The export map/JSDoc own the inventory. The bare barrel re-exports engine, standard and extras; `DEFAULT_PLUGINS` is only the zero-config subset. Extras star-exports author convenience/debug/import APIs, also on bare; opt-in engine capabilities (audio, mirror, bvh) stay in standard.

`runtime` owns platform/device services; `vite` is build-config tooling, never runtime. Published `*/core` subpaths own extension schemas, shader chunks, packing, tooling and diagnostics, not the author happy path. Consumers never deep-import source; missing public seams warrant an issue. Internal cross-module consumers use the subpath too (including utils codecs); standard uses relative paths, render producers use `render/core`.

Skin is engine-owned, format-independent pose storage, with no surface: glTF, physics and procedural producers compose `skin/core`. glTF decode/cache stays one-way, creating no entities; its plugin owns placement. Reuse the shared image-array upload and text atlas rather than reimplementing them. BVH stays rendering-unaware; its traversal TGSL closes over a consumer-named global and is not exported, only its raw splice is.

## Compiled tooling exports

Runtime exports ship raw TS for TypeGPU. Private `shallot-tooling` owns CLI/project/native sources. Its `scripts/build.ts` projects the public bin, source types, assets/crate and compiled `./vite`/`./harness/browser` defaults. One maintained source; no installed workspace reach. Build before linked imports and after postpack clears projections. Node-only islands exclude runtime/TGSL; the browser leaf is import-free.

The public raw `src/harness/index.ts` composes runtime plus the compiled browser leaf. Runtime's `harness/runtime.ts` and siblings never import that composite or tooling. Preserve the complete public value/type surface. Missing/stale projections fail; never infer ownership from the installed `src/` prefix.

CLI installs the project and TGSL plugins once. Ejected Vite uses direct `unplugin-typegpu/vite`, not CLI-only `typegpuPlugin`; pass the project directory to `projectPlugin(dir)` or it loads an empty manifest. Keep both engine and TypeGPU out of dependency prebundling; component TGSL runs after its framework with matching include IDs. Consumer recipes: `packages/shallot/MIGRATION.md`.

## Distribution layers

The installed package/CLI is canonical, usable with any framework/layout whose own bundler owns preview. Never trade it away for a convenience bundle. That bundle lives outside src, composes rather than duplicates the tool, and depends inward. Core remains unaware of it; headless manifest/plugin/scene resolution has no Vite/browser dependency.

## Barrel rules

Barrels expose author components, singletons, registration and types. Core subpaths expose extension internals; sibling-only implementation/test seams are neither. Keep sear codegen, glaze composite machinery beyond its chunk, and skin layout/CPU blend helpers internal. Preserve published compatibility; before removing an export, grep src AND every examples tier and read each call site for load-bearing use.

Every release-public export has definition-site JSDoc: lowercase, one line, examples on most callables. Fields must be actionable to scene authors, otherwise internal derived state; defaults are principled values, not float garbage. Avoid names needing import aliases: generic names belong in a subpath or stay internal.

List import-time effects AND barrels hosting bare effect imports in `sideEffects`, exact paths, no globs. Physics stays opt-in and tree-shakes away: Tumble/Physics on the main barrel, inlined tumble kernel with no extra install, AVBD only on its subpath. Backend install/uninstall and system anchors stay on `physics/core`, with no solver there. Curate `tumble/core` to mirror new engine public symbols but NEVER export shutdown: the shared worker pool is engine-owned. Init and read-only threads introspection remain, not threading knobs.

## Registry pattern

Use `Registry` directly; clear entries and ID space on rebuild. No pass-through wrappers.

## Type discipline

No primitive-renaming aliases or file-local exported types. Interfaces need multiple fields or methods.
