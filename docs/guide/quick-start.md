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

## Your First Scene

Shallot can be used with or without the built-in editor.

<!-- tabs -->
<!-- tab: UI -->

### Start the Editor

```bash
bunx shallot
```

![Editor layout](/captures/editor-layout/layout.webp)

The editor has three main areas: the **viewport** (center), the **outliner** (right, top), and the **inspector** (right, bottom). The demo scene starts with a camera, lights, and a ground plane.

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

coming soon

<!-- /tabs -->
