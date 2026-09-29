// The names of the engine's default plugins, kept as a dependency-free list so the `virtual:project`
// generator can classify a manifest without importing game modules into the Vite/Node tool process.
// `catalog.test.ts` gates it against the engine's real `DEFAULT_PLUGINS` so this list can't drift.
export const DEFAULT_PLUGIN_NAMES = [
    "Transforms",
    "Input",
    "Render",
    "Part",
    "Sear",
    "Glaze",
] as const;

// Engine plugins beyond the defaults, enabled by `name: true` and resolved from the
// main barrel (`import { OrbitPlugin } from "@dylanebert/shallot"`). Dep-free like the lists above and
// gated by catalog.test.ts against the barrel's real `*Plugin` exports so it can't drift. The toolchain
// warns on a `name: true` outside the union below (an unknown engine plugin, otherwise a cryptic esbuild
// "no export named ${name}Plugin" at bundle time).
export const EXTRA_PLUGIN_NAMES = [
    "Audio",
    "Character",
    "Fog",
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
    "Physics",
] as const;

/** every engine plugin name a manifest may enable with a bool — the union the toolchain validates against. */
export const KNOWN_ENGINE_PLUGINS: ReadonlySet<string> = new Set<string>([
    ...DEFAULT_PLUGIN_NAMES,
    ...EXTRA_PLUGIN_NAMES,
]);
