// The names of the engine's default plugins, as a dep-free string list. The `virtual:project` generator
// runs in the vite/Node process (no WebGPU globals — importing a plugin object evaluates GPU-touching
// module code like sear's top-level `GPUShaderStage`), so it can't read the names off the plugin objects;
// it classifies a manifest entry as a default vs an extra/local from this list alone. `catalog.test.ts`
// gates it against the engine's real `DEFAULT_PLUGINS` so this list can't drift.
export const DEFAULT_PLUGIN_NAMES = [
    "Slab",
    "Transforms",
    "Input",
    "Render",
    "Part",
    "Sear",
    "Glaze",
] as const;

// Engine plugins that ship on their own subpath rather than the main barrel (exports.md — a backend
// plugin like `AvbdPlugin` is barrel-adjacent, not barrel-listed). Dep-free for the same reason as
// DEFAULT_PLUGIN_NAMES above; catalog.test.ts gates each entry against the real subpath export.
export const SUBPATH_PLUGIN_MODULES: Record<string, string> = {
    Avbd: "@dylanebert/shallot/avbd",
};

// Engine plugins beyond the defaults + subpath backends, enabled by `name: true` and resolved from the
// main barrel (`import { OrbitPlugin } from "@dylanebert/shallot"`). Dep-free like the lists above and
// gated by catalog.test.ts against the barrel's real `*Plugin` exports so it can't drift. The toolchain
// warns on a `name: true` outside the union below (an unknown engine plugin, otherwise a cryptic esbuild
// "no export named ${name}Plugin" at bundle time).
export const EXTRA_PLUGIN_NAMES = [
    "Audio",
    "Character",
    "Fog",
    "Gltf",
    "Lines",
    "Mirror",
    "Orbit",
    "OrbitOverlay",
    "Outline",
    "Player",
    "Profile",
    "Sky",
    "Sprite",
    "Text",
    "Tumble",
    "Tween",
] as const;

/** every engine plugin name a manifest may enable with a bool — the union the toolchain validates against. */
export const KNOWN_ENGINE_PLUGINS: ReadonlySet<string> = new Set<string>([
    ...DEFAULT_PLUGIN_NAMES,
    ...EXTRA_PLUGIN_NAMES,
    ...Object.keys(SUBPATH_PLUGIN_MODULES),
]);
