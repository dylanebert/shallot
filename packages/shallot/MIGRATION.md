# Migrating from 0.8 to 0.9.2

This port touches GPU code only: the ECS, scene, and ordinary component APIs keep their 0.8 shape. What moves is the GPU substrate. Layouts now come from TypeGPU schemas, custom shaders use TGSL, and render registries carry typed resources.

Port to 0.9.2, the newest patch. The 0.9 API is the same across all three, and the patches carry the fixes a port hits first: compiled tooling exports, a duplicate-TypeGPU check that catches two copies of the same pinned minor, and the TypeGPU 0.12 toolchain the install block below pins. [`CHANGELOG.md`](https://github.com/dylanebert/shallot/blob/main/CHANGELOG.md) has the list.

Already on 0.9.0 or 0.9.1? You need none of this guide: only the TypeGPU toolchain bump in that install block, since 0.9.2 moved the peer to the 0.12 minor.

## Install the GPU toolchain

Install the engine and its TypeGPU peer at the same time. Keep TypeGPU on the 0.12 minor used by Shallot; two copies in one bundle race over the same metadata map. The TypeGPU plugin versions move with the library, never independently.

```bash
bun add @dylanebert/shallot@^0.9.2 typegpu@~0.12.0
bun add -d unplugin-typegpu@~0.12.1 eslint@^9 eslint-plugin-typegpu@~0.12.0
bun add -d @babel/core@^7.28.6 @babel/eslint-parser@^7.28.6 @babel/plugin-syntax-typescript@^7.28.5
```

A manifest project with no `vite.config` is done. `shallot dev` and `shallot build` synthesize a config with the transform included.

An ejected Vite project adds one TypeGPU plugin, plus the dev-mode dependency exclusion:

```ts
// vite.config.ts
import { defineConfig } from "vite";
import typegpu from "unplugin-typegpu/vite";
import { projectPlugin } from "@dylanebert/shallot/vite";

export default defineConfig({
    plugins: [typegpu(), projectPlugin(".")],
    optimizeDeps: { exclude: ["@dylanebert/shallot", "typegpu"] },
});
```

`@dylanebert/shallot/vite` ships compiled as of 0.9.1, which is what lets a config file import it at all. Vite's default config loader bundles `vite.config.ts` and runs it in a real `node` subprocess, and Node applies no TypeScript transform to a `node_modules` import, so against 0.9.0 this recipe throws `ERR_UNKNOWN_FILE_EXTENSION` before Vite starts. `@dylanebert/shallot/harness/browser` in a `playwright.config.ts` is the same shape.

The direct `unplugin-typegpu/vite` plugin is the supported ejected-project route. `typegpuPlugin()` is reserved for Shallot's synthesized CLI config; do not add it here. A second transform rewrites generated metadata and breaks it.

`projectPlugin`'s argument is the directory it reads `shallot.json` (plugins + scene) from — omit it and `virtual:project` resolves to an empty manifest (default plugins, no scene, nothing to render), the same degenerate shape `shallot dev`/`shallot build` never produce because they always pass the project directory.

Keep the `optimizeDeps` exclusion, and name both packages. The bare `@dylanebert/shallot` specifier covers every one of its subpath imports, but `typegpu` needs its own entry whenever your own source imports `typegpu/data`, or any other typegpu subpath, directly: that is a second, independent scanner entry the engine's exclusion doesn't reach.

Why: a registry install resolves both packages inside `node_modules`, so Vite's dev-server dependency scanner esbuild-prebundles them ahead of the typegpu transform on first page load. No Vite plugin, `typegpu()` included, runs over a prebundled dependency, and the result is a duplicate, untransformed TypeGPU identity that dies at pipeline warm.

The line is load-bearing on Shallot's own zero-config path (`shallot dev` / `shallot build`'s synthesized config, red-proven by removing it). On this ejected recipe it is a defensive default rather than a proven requirement: removing it from this exact on-disk `vite.config.ts` boot has not reproduced the failure across repeated real-hardware runs, and why that boot path differs from the CLI's synthesized one is unexplained.

### Svelte and Vue

TypeGPU's default include matches JavaScript and TypeScript files. A TGSL function written inside a `.svelte` or `.vue` script block sits outside that filter after the framework transform. Run TypeGPU after the framework plugin and include the compiled component id:

```ts
import typegpu from "unplugin-typegpu/vite";

const tgslFiles = [
    /\.m?[jt]sx?(?:\?.*)?$/,
    /\.svelte(?:\?.*)?$/,
    /\.vue(?:\?.*)?$/,
];

export default defineConfig({
    plugins: [framework(), typegpu({ enforce: "post", include: tgslFiles })],
    optimizeDeps: { exclude: ["@dylanebert/shallot", "typegpu"] },
});
```

The `optimizeDeps` exclusion above still applies here — a framework plugin changes which files the transform reaches, not whether the dev-server dependency scanner can prebundle ahead of it.

Keep the one component extension your project uses. `checkTgsl()` runs when Shallot requests a GPU, but it checks engine `.ts` metadata only. Add a consumer test that resolves or CPU-calls one TGSL function from your component.

### Plain Bun and Node

Bun needs the plugin registered from a preload. The Bun adapter exports a factory and accepts one regular expression:

```ts
// tgsl-preload.ts
import { plugin } from "bun";
import typegpu from "unplugin-typegpu/bun";

plugin(typegpu({ include: /\.tsx?$/ }));
```

```toml
# bunfig.toml
preload = ["./tgsl-preload.ts"]
```

Keep the filter on TypeScript. A broad JavaScript filter also reloads pruned dependencies and can strip CommonJS default-export interop.

Node has no preload adapter. Bundle the entry through a supported adapter such as esbuild, then run the emitted module:

```bash
bun add -d esbuild
```

```ts
// build.ts
import { build } from "esbuild";
import typegpu from "unplugin-typegpu/esbuild";

await build({
    entryPoints: ["src/main.ts"],
    bundle: true,
    format: "esm",
    platform: "node",
    outfile: "dist/main.mjs",
    plugins: [typegpu()],
});
```

Run `bun build.ts && node dist/main.mjs`. Node cannot execute the source directly because TypeGPU has no runtime parser.

## Replace handwritten layouts with schemas

In 0.8, a buffer binding repeated its WGSL type as a string:

```ts
const bindings = {
    particles: { type: "storage", element: "Particle", access: "read_write" },
};
```

In 0.9, the schema is the layout:

```ts
import * as d from "typegpu/data";

export const Particle = d
    .struct({ pos: d.vec4f, velocity: d.vec4f })
    .$name("Particle");

const bindings = {
    particles: { type: "storage", element: Particle, access: "read_write" },
};
```

Name public and factory-built schemas with `.$name()`. A schema owns field order, alignment, packing, and the emitted WGSL declaration. Delete the duplicate WGSL struct after every consumer uses the schema.

Type-check storage texture formats. A misspelled format can resolve to `texture_storage_2d<undefined, ...>` without a TypeGPU resolve error. Do not widen a format to `string` or cast it through `GPUTextureFormat`; keep the closed `StorageTextureFormats` type at the call.

## Move buffers to typed handles

Create a typed buffer from its schema, or wrap a raw allocation without changing ownership:

```ts
const ParticleArray = d.arrayOf(Particle, count);

const owned = Compute.root.createBuffer(ParticleArray).$usage("storage");
const wrapped = Compute.root.createBuffer(ParticleArray, rawBuffer).$usage("storage");
const rawAgain = Compute.root.unwrap(wrapped);
```

Use `Compute.typed` for a named typed buffer and `Compute.buffers` for its raw twin when old or external code still needs the `GPUBuffer`. The allocation owner destroys it. A wrapper over a sibling's buffer is borrowed.

`Draw.args.indirect`, mesh vertex/index/storage fields, `Parts.drawArgs`, and typed Mirror sources now preserve their schema and usage flags. Replace broad `GPUBuffer` annotations and casts with the exported typed contract. `Mirror` accepts either a typed or raw buffer.

Keep per-frame CPU data in typed arrays or `ArrayBuffer`, then write that memory. Do not build arrays of schema-shaped objects in a hot path.

## Port custom shaders to TGSL

Replace a 0.8 WGSL string with a TypeGPU function:

```ts
const integrate = tgpu.fn([Particle, d.f32], Particle)((particle, dt) => {
    "use gpu";
    return Particle({
        pos: particle.pos + particle.velocity * dt,
        velocity: particle.velocity,
    });
});
```

A pure `tgpu.fn` runs on the CPU and resolves into the GPU function, so use it as the unit-test truth. Use WGSL-bodied `tgpu.fn` leaves only for constructs TypeGPU 0.11 cannot express. The raw splice surface remains available through the `*Wgsl()` exports under `*/core`.

Porting rules that catch silent wrong code:

- initialize integer locals with `d.u32(...)` or `d.i32(...)`; a bare integer literal is `i32`
- use `idiv` from `@dylanebert/shallot/utils/core`; `/` on integer operands emits float division
- copy an array element through its schema before assigning or returning it; TypeGPU can represent an element read as a pointer
- name every factory-built kernel and pipeline with `.$name()`
- add `eslint-plugin-typegpu`'s recommended config and fail on warnings

Shallot itself uses this flat ESLint config:

```ts
import parser from "@babel/eslint-parser";
import typegpu from "eslint-plugin-typegpu";

export default [
    { ignores: ["**/*.d.ts"] },
    {
        ...typegpu.configs.recommended,
        files: ["**/*.ts", "**/*.tsx"],
        languageOptions: {
            parser,
            parserOptions: {
                requireConfigFile: false,
                babelOptions: { plugins: ["@babel/plugin-syntax-typescript"] },
            },
        },
    },
];
```

Run it as `eslint . --max-warnings=0`. Use your existing TypeScript ESLint parser instead when it supports your TypeScript version.

Two things to expect on the 0.12 toolchain, both from one change. TypeGPU deprecates `>>` on a `u32` operand in favor of `>>>`. JS's `>>` is arithmetic and WGSL's on `u32` is logical, so a CPU-callable TGSL function and the shader it generates disagree on any word with bit 31 set. Both forms emit WGSL `>>`, so switching changes no shader.

The plugin also stopped flagging the syntax those shifts used to trip. An `eslint-disable typegpu/no-unsupported-syntax` you added for one is now an *unused* directive, which `--max-warnings=0` fails on. Drop that rule from the directive and keep any other rule it disabled alongside.

## Port custom surfaces and backgrounds

The 0.8 surface contract accepted `bindings`, `preamble`, `interpolators`, `vs`, and `fs` strings. The 0.9 contract separates layout creation from State-owned registration, then takes TGSL functions:

```ts
import tgpu from "typegpu";
import * as d from "typegpu/data";
import {
    fsCtxSchema,
    registerSurface,
    surfaceLayout,
} from "@dylanebert/shallot/sear/core";

const Tint = d.struct({ color: d.vec4f }).$name("Tint");
const layout = surfaceLayout({ tint: { type: "uniform", struct: Tint } });
const fs = tgpu.fn([fsCtxSchema()], d.vec4f)((ctx) => {
    "use gpu";
    return layout.$.tint.color;
});

registerSurface(state, { name: "tinted", layout, fs });
```

Create `layout` before `fs` or `vs`; the function closes over `layout.$.bindingName`. Use `registerSurface(state, spec)` instead of `Surfaces.register(spec)` so disposal cannot remove a later State's replacement.

For custom varyings, define one schema map and pass it to both sides:

```ts
const varyings = { tint: d.vec3f };
const vs = tgpu.fn([VsIn], vsPatchSchema(varyings))((input) => { /* ... */ });
const fs = tgpu.fn([fsCtxSchema(varyings)], d.vec4f)((ctx) => { /* ... */ });
```

List built-in mesh fields read by `fs` under `fragmentInputs` (`uv` and/or `localPos`). Undeclared fields stay available to `vs`, but become zero in the fragment context and consume no interpolator. `blend`, `screen`, and compile-time `specialize(variant)` keep their 0.8 roles.

Backgrounds use the same pattern with `backgroundLayout`, `BgCtx`, and `registerBackground(state, spec)`.

## Replace changed shader exports

Relocatable shader text is now usually resolved lazily from the same TGSL functions the CPU tests call. Every shader and fog-oracle helper exported by the documented 0.8.1 package subpaths that changed name or shape:

| 0.8 | 0.9 |
| --- | --- |
| `FRAME_STRUCT_WGSL` | `frameWgsl()` and `FrameGpu` |
| `VIEW_STRUCT_WGSL` | `viewWgsl()` and `View` |
| `LIGHTING_STRUCT_WGSL` | `lightingWgsl()` and `LightingGpu` |
| `POINT_LIGHTS_STRUCT_WGSL` | `pointLightsWgsl()` and `PointLights` |
| `LINEAR_TO_SRGB_WGSL` | `linearToSrgbWgsl()` |
| `TONEMAP_WGSL` | `tonemapWgsl()` |
| `FOG_STRUCT_WGSL` | `fogStructWgsl()` and `FogGpu` |
| `FOG_MARCH_WGSL` | `fogMarchWgsl()` |
| `FOG_INSCATTER_WGSL` | `fogInScatterWgsl()` |
| `FogLight` | removed; `fogInScatter` now accepts `d.Infer<typeof PointLightGpu>`, inferred from the engine schema exported by `render/core` |
| `reconstruct(invViewProj, u, v, depth)` | `reconstructWorld(invViewProj, d.vec2f(u, v), depth)`; the matrix and vector use TypeGPU data values and the TGSL function remains CPU-callable |
| `LIGHT_EVAL_WGSL` | `lightEvalWgsl()` |
| `SUN_SHADOW_STRUCT_WGSL` | `sunStructWgsl()` and `SunShadow` |
| `SAMPLE_SUN_SHADOW_WGSL` | `sunShadowWgsl()` and `sampleSunShadow` |
| `COLLIDE_WGSL` / `HULL_WGSL` | `collideWgsl()` / `hullWgsl()` |
| `BVH_ROOT_WGSL` / `BVH_TRAVERSE_WGSL` | `bvhRootWgsl()` / `bvhTraverseWgsl()` |
| `OCT_ENCODE_WGSL` | `octEncodeWgsl()` |
| `POS_QUANT_WGSL` / `POS_QUANT_PACK_WGSL` | `posQuantWgsl()` / `posQuantPackWgsl()` |
| `XFORM_WGSL` | `xformWgsl()` |
| `LDR_COLOR_UNPACK_WGSL` | `ldrColorUnpackWgsl()` |
| `pack2x16unorm` / `unpack2x16unorm` | `packUnorm2x16` / `unpackUnorm2x16` |

`LINEAR_TO_OKLAB_WGSL` and `OKLAB_TO_LINEAR_WGSL` stayed exported raw constants through 0.9.2 and became `linearToOklabWgsl()` and `oklabToLinearWgsl()` in 0.9.3, joining the rest of this table; a port targeting 0.9.3 calls the thunks, and no alias under the old names exists. The already-lowercase `casterWgsl()` and `pointShadowWgsl()` names also remain. Call each new thunk when composing raw WGSL. Do not resolve a chunk at module import time; resolution needs the active feature set and shared namespace.

## Warm typed pipelines

TypeGPU 0.11 creates pipelines synchronously, and Dawn can defer the real compile to the first dispatch. Register a zero-workgroup force during `warm`:

```ts
precompile("particles", () => {
    const bound = bindForWarm();
    bound.dispatchWorkgroups(0);
    return bound;
});
```

The label must be unique in a build. Allocate and bind throwaway inputs inside the callback, which runs after plugin warms. Return the non-null bound pipeline after dispatching; a missing return fails instead of moving the stall to frame one. Use `options.after` when a producer publishes a resource during another warm.

## Keep raw WebGPU where it owns the boundary

0.9 still adopts an external `GPUDevice`, wraps an external `GPUBuffer`, unwraps typed layouts, bind groups, buffers, and pipelines, and resolves typed functions into raw WGSL. Shallot continues to own encoders and render passes, so a TypeGPU pipeline can execute inside an existing pass through `.with(pass)`.

The repository's [`examples/flows/no-walls`](https://github.com/dylanebert/shallot/tree/main/examples/flows/no-walls) flow exercises every direction. Use those escapes at integration boundaries. Keep engine-authored kernels in TGSL so their schemas, CPU tests, lint rules, and diagnostics remain active.

## Verify the port

Run these from the project root:

```bash
bunx tsc --noEmit
bunx eslint . --max-warnings=0
bunx shallot build
bunx shallot verify --dist
```

`shallot verify` uses Shallot's optional Playwright peer. Before the first verify run, install it and its Chromium binary:

```bash
bun add -d playwright
bunx playwright install chromium
```

Resolve or CPU-call one consumer-owned TGSL function. `requestGPU` checks engine metadata but cannot prove a `.svelte` or `.vue` block passed through your filter. For a custom GPU path, keep a final resource or framebuffer assertion in `shallot verify`; emitted WGSL alone does not prove WebGPU accepted or consumed it.
