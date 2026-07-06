---
title: Quick Start
description: build your first scene
icon: rocket
order: 0
---

## Setup

### Install [Bun](https://bun.sh)

<!-- os -->
<!-- os: Windows -->

```bash
powershell -c "irm bun.sh/install.ps1 | iex"
```

<!-- os: Mac/Linux -->

```bash
curl -fsSL https://bun.sh/install | bash
```

<!-- /os -->

### Create a Project

```bash
bun create shallot my-game
cd my-game
bun install
```

This scaffolds a minimal project: a `shallot.json` manifest, a plugin, and a scene. Run it with the CLI:

<!-- pick -->
<!-- pick: Editor -->

Open the project in the visual editor.

```bash
bunx shallot
```

<!-- pick: Dev -->

Run it standalone with hot reload.

```bash
bunx shallot dev
```

<!-- pick: Build -->

Ship a web bundle. Add `--target windows|mac|linux` for a native build.

```bash
bunx shallot build
```

<!-- /pick -->

### The manifest

`shallot.json` is the project's source of truth: the scene plus which plugins are on. The editor, `shallot dev`, and `shallot build` all read it identically.

```json
{
  "$schema": "./node_modules/@dylanebert/shallot/shallot.schema.json",
  "scene": "scenes/scene.scene",
  "plugins": {
    "Orbit": true,
    "Spin": "./src/spin"
  }
}
```

Each `plugins` entry maps a name to a source. `true` / `false` toggles an **engine** plugin, resolved by name from `@dylanebert/shallot`. A **module specifier** declares a local or installed plugin, whose **default export** is the Plugin: a relative path (`./src/spin`, your own code) or a package subpath (`my-pack/widget`, a plugin library).

To add a plugin from anywhere, install it with your package manager (`bun add a-pack`, a git URL, a registry) and list its subpath here. The editor toggles these on and off and writes the change back: a disabled local becomes `["./src/spin", false]`, the `[source, enabled]` tuple form that keeps the spec. The `$schema` line gives autocomplete and validation as you edit the JSON.

## Your First Scene

Shallot can be used with or without the built-in editor.

<!-- tabs -->
<!-- tab: Editor -->

### Start the Editor

```bash
bunx shallot
```

![Editor layout](/captures/editor-layout/layout.webp)

The editor has three main areas: the **outliner** (left), the **viewport** (center), and the **inspector** (right). The scaffolded scene starts with a camera, lights, and a cube.

### Selecting Entities

Click an entity in the outliner to select it. The inspector shows its components.

![Ground selected in outliner](/captures/select-entity/selected.webp)

### Editing Properties

Select an entity and edit its fields in the inspector. Changes apply immediately.

![Editing transform position](/captures/edit-transform/transform-edited.webp)

### Adding Entities

Click **Add Entity** at the bottom of the outliner to create a new entity.

![New entity added](/captures/add-entity/after.webp)

### Adding Components

Click **Add Component** in the inspector to attach new behavior. Type to filter the list.

![Component picker](/captures/add-component/picker-open.webp)

### Play Mode

Click the play button in the toolbar to run the scene. Click stop to return to editing.

![Scene playing](/captures/play-mode/playing.webp)

<!-- tab: Code -->

Prefer to build in code? Every plugin and scene the editor writes is plain TypeScript and XML you can author by hand. Start with [ECS](doc:engine/ecs) to write a system, the behavior the editor can't author for you, then the [Scene](doc:engine/scene) format for the world file. Each reference page opens with a scene snippet, then the code that drives it.

<!-- /tabs -->
