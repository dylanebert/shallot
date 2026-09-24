# Migrating from 0.9.5 to 0.10

Most 0.9.5 game code carries over. These changes break it.

## `shallot recipe` is now `shallot add`

```sh
# 0.9.5
bunx shallot recipe first-person

# 0.10
bunx shallot add first-person
```

## `shallot tui` and `shallot verify` are gone

Neither has a replacement command. `@dylanebert/shallot/harness/check` exports `check()` for writing checks as Bun tests; it does not launch a browser.

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

## `/harness/browser` no longer resolves

0.10 ships no browser-launch configuration. Configure the browser in your own test setup.

## TypeGPU below 0.12.5 is too old

```sh
bun add typegpu@~0.12.5
bun add -d unplugin-typegpu@~0.12.3
```

`shallot dev` and `shallot build` add the TypeGPU compiler plugin. A project with its own Vite config adds it once, even if it only uses Shallot's shaders:

```ts
import typegpu from "unplugin-typegpu/vite";
import { projectPlugin } from "@dylanebert/shallot/vite";

export default defineConfig({ plugins: [typegpu(), projectPlugin(".")] });
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
