// Part's extension surface: the pack's GPU-output registry. `Part` (the component) + `Color` ride the main
// barrel; `Parts` is the internal pack output — the slot-major `drawArgs` (DrawIndexedIndirect) + packed
// survivor eids a custom pipeline or a GPU-readback oracle reads. GPU handles, not author API, so it lives
// at the extension tier like render's `Draws` / `Surfaces` registries.

export { Parts } from "./part";
