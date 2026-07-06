// The names of the engine's default plugins, as a dep-free string list. The `virtual:project` generator
// runs in the vite/Node process (no WebGPU globals — importing a plugin object evaluates GPU-touching
// module code like sear's top-level `GPUShaderStage`), so it can't read the names off the plugin objects;
// it classifies a manifest entry as a default vs an extra/local from this list alone. The editor client,
// which has the objects, derives the same names from them — `catalog.test.ts` gates the two against the
// engine's real `DEFAULT_PLUGINS` so this list can't drift.
export const DEFAULT_PLUGIN_NAMES = [
    "Slab",
    "Transforms",
    "Input",
    "Render",
    "Part",
    "Sear",
    "Glaze",
] as const;
