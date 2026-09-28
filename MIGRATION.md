# Migrating from 0.9.5 to 0.10

Most 0.9.5 game code carries over. These changes break it.

## `build()` always requires WebGPU

Every app now acquires a GPU device during `build()`, or rejects with the acquisition cause. A composition no longer runs as a CPU-only app. Supply an already-acquired device with `config.device` when the host owns acquisition; otherwise the engine uses `navigator.gpu`.

`Plugin.device` is removed: plugins no longer declare `required` or `optional`, and `deviceTier` / `DeviceTier` are gone. `warmPlugins` also requires a `GPUDevice` and has no CPU-tier argument. `resetCompute`, which only cleared the device for CPU builds, is removed. The device-tier report, generator, checker, its test and fixtures, and the Device tiers section of `CONTRIBUTING.md` are removed with no replacement. The import-time WebGPU enum shim `project/gpu-globals.ts` is removed because engine modules no longer read WebGPU globals during import.

Device acquisition no longer requests adapter-reported maxima wholesale. It requests Shallot's fixed storage-buffer floor and compatibility values where split-stage limits are unsupported; otherwise the device keeps its default limits. Operations that exceed a limit refuse at their own named check.

Tests that build an app use `*.gpu.test.ts`; the root Bun test preload applies Shallot's TypeGPU transform. `bun run test` includes these tests on a device, while hosted jobs without one exclude the suffix and the macOS GPU job selects it with `bun test gpu.test`. Node tests and manual oracles keep their existing tiers. The CPU-only first-person Node allocation sample and packed-install CPU-build smoke are removed; there is no CPU app to measure or build. The acquisition-refusal test replaces adapter acquisition with a named failure, so it does not need a real adapter.

## `shallot recipe` is now `shallot add`

```sh
# 0.9.5
bunx shallot recipe first-person

# 0.10
bunx shallot add first-person
```

## The CLI command set changed

- `shallot list` and `shallot check` are removed. Run the project's cheap tests with `bun test`, named host tiers by file path, and browser tests with `bunx playwright test`.
- `shallot workflow` has no replacement; the workflow file it wrote is yours to edit and is no longer regenerated.
- `shallot <verb>` no longer resolves to `shallot-<verb>` on `PATH`; there is no replacement.
- `shallot run` is now `shallot build` followed by `shallot preview`. Preview never rebuilds.

## `shallot tui` and `shallot verify` are gone

Neither has a replacement command. Tests use Bun's `test()` directly; browser tests use Playwright Test against the project's Vite preview.

## `/render/core`, `/sear/core` and `/utils/core` no longer resolve

Package imports dropped their `/core` suffix, and rendering split in two: `/rendering` for what every renderer shares, and `/standard/rendering` for Shallot's default mesh renderer.

```ts
// 0.9.5
import { FrameGpu } from "@dylanebert/shallot/render/core";
import { engineLayout, registerSurface, surfaceLayout } from "@dylanebert/shallot/sear/core";
import { Xform } from "@dylanebert/shallot/utils/core";

// 0.10
import { FrameGpu, registerSurface, surfaceLayout } from "@dylanebert/shallot/rendering";
import { engineLayout } from "@dylanebert/shallot/standard/rendering";
import { Xform } from "@dylanebert/shallot/utils";
```

Likewise `/ecs/core` is `/ecs`, `/scene/core` is `/scene` and `/physics/core` is `/physics`. The `/src/*` wildcard is gone: only the paths in `package.json` `exports` resolve.

## `Inputs` is now `devices(state)`

Input is plain data on each State: `devices(state)` returns its keys, mouse, touch and viewport. With the default plugins the browser fills it; a test or replay fills the same record with `pressKey`, `pointerMove` and the other producers, with no browser. `setInputEnabled` takes the State, and the canvas size moved from `mouse` to `viewport`.

```ts
// 0.9.5
if (Inputs.isKeyDown("KeyW")) moveForward();
if (Inputs.isKeyPressed("Space")) jump();
const width = Inputs.mouse.canvasWidth;
setInputEnabled(false);

// 0.10
const input = devices(state);
if (input.keys.held.has("KeyW")) moveForward();
if (input.keys.pressed.has("Space")) jump();
const width = input.viewport.get(input.focused)?.cssWidth ?? 0;
setInputEnabled(state, false);

// a test drives the same record
pressKey(state, "KeyW");
```

`keys.pressed` holds a press until the next frame. A `fixed` system reads `keys.tickPressed`, which holds it until the next fixed update.

`isKeyPressedWithin(code, seconds)` measured wall time, so its replacement counts fixed updates: `keys.pressedTick` records the update each key was pressed on.

```ts
const at = input.keys.pressedTick.get("Space");
if (at !== undefined && state.time.fixedTick - at < 6) jump(); // pressed within the last 6 updates
```

## `Tumble` is now `Physics`, and `Physics.backend` is gone

The plugin, its manifest key and its import path are renamed. Bodies are read and driven by functions from `/physics` that take the State.

```json
// 0.9.5
{ "plugins": { "Tumble": true } }

// 0.10
{ "plugins": { "Physics": true } }
```

```ts
// 0.9.5
Physics.backend?.setKinematic(eid, position, rotation);
const body = Tumble.body(eid);

// 0.10
import { body, setKinematic } from "@dylanebert/shallot/physics";
setKinematic(state, eid, position, rotation);
const b = body(state, eid);
```

`Tumble.world` is `physicsWorld(state)`, and `readBody(state, eid)` reads a body's pose.

## `/avbd` is gone

0.10 ships no AVBD solver. Apps that select it use the built-in `Physics` instead.

## `Tween`, `Sequence` and `/tween/core` are gone

0.10 has no animation plugin. Animation is app code.

## `/document` and edit mode are gone

`Document`, `History`, `Session` and `ReadbackSystem` are removed, with no replacement for undo, redo or editor sessions. `State.mode`, the app's `mode` option and `annotations.mode` are removed too: every system always runs. Saving and loading scenes still works:

```ts
import { serialize, stringify } from "@dylanebert/shallot";
const saved = stringify(serialize(state));
```

## Browser tests use Playwright Test

Configure Playwright in the project, run its own Vite preview with `webServer`, and put Chromium launch flags in `playwright.config.ts`.

## TypeGPU below 0.12.6 is too old

```sh
bun add typegpu@~0.12.6
```

For web, `shallot dev`, `shallot build` and `shallot preview` run the project's Vite commands. Native `dev` and `build` add the desktop shell to the Vite server or build; native `preview` launches that build. Add Shallot's Vite plugin to the project config; it includes the TypeGPU transform, so do not register a separate TypeGPU plugin:

```ts
import { shallot } from "@dylanebert/shallot/vite";

export default defineConfig({ plugins: [shallot()] });
```

## `sparse(u8)` values wrap

Single-value `sparse` fields now store their declared type on every write: `sparse(u8)` stores 300 as 44, `sparse(u32)` stores negatives as unsigned, and `sparse(f32)` rounds to 32-bit precision. Declare a type wide enough for the values you store.

## `tsc` reports missing `ImportMeta.env` or Node types

The 0.9.5 scaffold's `tsconfig.json` lists only WebGPU types. Add Node and Vite's:

```sh
bun add -d @types/node@^26.0.0
```

```json
{ "compilerOptions": { "types": ["@webgpu/types", "node", "vite/client"] } }
```

These are types only; no Node code reaches the browser bundle.
