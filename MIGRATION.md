# Migration

## 0.10

Device reads are State-scoped: replace the removed `Inputs` singleton with `devices(state)` and read its `keys`, `mouse`, `touch`, `focused`, `viewport` or `audio` rows.

- `isKeyPressedWithin` was removed; use `devices(state).keys.pressedTick` with a fixed-tick window such as Character's jump and coyote timers.
- `Mouse.canvasWidth` and `Mouse.canvasHeight` were removed; use `devices(state).viewport.get(devices(state).focused)` for the focused canvas's CSS size.
- `setInputEnabled` takes the State explicitly: `setInputEnabled(state, on)`.
