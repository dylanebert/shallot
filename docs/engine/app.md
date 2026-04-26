---
title: App
description: starting your game, adding features
source: engine/app
icon: play
order: 2
---

# App

<!-- tabs -->
<!-- tab: UI -->

coming soon

<!-- tab: Code -->

`run()` loads plugins, creates entities from a scene file, and starts the game loop.

```typescript
import { run, OrbitPlugin } from "@dylanebert/shallot";

const state = await run({
    plugins: [OrbitPlugin, MyPlugin],
    scene: "/scenes/demo.scene",
});
```

## Config

| Field | Type | Required | Description |
|---|---|---|---|
| `plugins` | `Plugin[]` | yes | plugins to load, merged with built-in defaults |
| `scene` | `string \| string[]` | no | `.scene` file path, array of paths, or inline XML string |
| `defaults` | `boolean` | no | `false` to skip built-in plugins |
| `exclude` | `Plugin[]` | no | specific built-in plugins to remove |
| `setup` | `(state: State) => void` | no | runs after registration, before plugin init |
| `loading` | `Loading` | no | loading screen (see below) |
| `ui` | `(el: HTMLElement, state: State) => () => void` | no | control panel overlay (see below) |

`plugins` is the only required field. An empty `[]` is valid (uses defaults only). Inline XML works for `scene` — any string starting with `<` is parsed directly.

## Lifecycle

`build()` runs the full startup sequence. `run()` calls `build()`, mounts UI, and starts the loop.

### 1. Collect Plugins

Default plugins load first (rendering, input, transforms, etc.), then `config.plugins`. `defaults: false` skips defaults. `exclude` removes specific ones. Each plugin's `dependencies` are checked — missing dependencies log a warning and the plugin is fully skipped (no registration, initialization, or warm).

### 2. Register

All plugins are iterated in insertion order. For each plugin, components, relations, and systems are registered on State. Nothing executes yet.

### 3. config.setup()

Runs before any plugin code executes. Use for setting resources or pre-configuring State.

### 4. Initialize Plugins

Plugins are topologically sorted by their `dependencies`. `plugin.initialize()` is called on each in dependency order — dependencies first, dependents after. No scene entities exist yet. This is where plugins create GPU pipelines, allocate buffers, and set up internal state.

### 5. Load Scenes

Scene files are fetched, parsed from XML, diagnosed for warnings, and loaded. This creates all the entities and components declared in the scene. After this step, the world is populated.

### 6. Warm Plugins

`plugin.warm()` runs on all plugins **in parallel**. Entities are available. Use for work that depends on scene content: building acceleration structures, precomputing spatial data, compiling shader variants for materials that now exist.

### 7. Game Loop

`run()` starts the frame loop after build. Each frame calls `state.step(dt)`, which runs systems in four groups:

| Group | Timing | Use |
|---|---|---|
| `setup` | once per frame | first-time system initialization, per-frame setup |
| `fixed` | 1-4 steps at 1/60s | physics, deterministic simulation |
| `simulation` | once per frame, variable dt | gameplay, animation, transforms |
| `draw` | once per frame, after simulation | rendering, GPU dispatch |

Systems within a group are ordered by `first`, `last`, `before`, `after` constraints, topologically sorted. A system's `setup()` runs once on its first frame, then `update()` runs every applicable frame.

Double-buffered GPU fences provide backpressure — at most 2 frames pending. Compute owns pacing via `sync()` on the Compute resource.

## build vs run

`build()` runs steps 1-6 and returns State. `run()` calls `build()`, mounts the UI overlay (web only), then starts the frame loop.

```typescript
const state = await build(config);
// manual control: inspect state, add entities, drive your own loop
state.step(dt);
```

## Default Plugins

```typescript
await run({ plugins: [MyPlugin], exclude: [PhysicsPlugin] }); // skip specific defaults
await run({ plugins: [MyPlugin], defaults: false });           // skip all defaults
```

`setDefaultPlugins()` and `setDefaultLoading()` configure what the defaults are.

## UI Overlay

The `ui` callback receives a container and State. The container overlays the canvas with `pointer-events: none`.

```typescript
ui: (container, state) => {
    const panel = document.createElement("div");
    panel.style.pointerEvents = "auto";
    container.appendChild(panel);
    return () => panel.remove();
}
```

Must return a cleanup function, called on `state.dispose()`.

## Loading Screen

```typescript
interface Loading {
    show(): (() => void) | void;
    update(progress: number): void;
    error?(message: string): void;
}
```

`show()` displays the loading UI and optionally returns a cleanup function. `update()` receives 0-1 progress across all lifecycle steps (initialize, scene load, warm). `error()` displays a fatal error on the loading overlay when initialization fails. `NoLoading` is a no-op implementation.

## Runtime

Platform layer in `@dylanebert/shallot/runtime`:

```typescript
import { Runtime, now, requestFrame, readFile } from "@dylanebert/shallot/runtime";

Runtime;                       // "web" | "headless"
now();                         // performance.now()
requestFrame(callback);        // rAF on web, setTimeout on headless
await readFile("/scene.scene"); // fetch on web, Bun.file on headless
```

## HMR

Dispose State on hot reload to prevent stacking game loops:

```typescript
if (import.meta.hot) {
    import.meta.hot.dispose(() => state.dispose());
}
```


<!-- tab: Reference -->

<!-- API:engine/app -->

<!-- /tabs -->
