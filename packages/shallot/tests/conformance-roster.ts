// the shared roster and rebuild-conformance harness, read by both the default-tier
// `conformance.test.ts` and the by-path `conformance.tier.ts`. One roster, split by a declared
// predicate (`isPipelineCompiling`), never two hand-copied lists — the one-source-of-truth law
// (`coding.md` "One source of truth"); a restated roster trips `check-docs.ts`'s tier-suffix arm.
//
// See `conformance.test.ts`'s header for the reload-conformance contract this harness pins.

import {
    build,
    type Plugin,
    type State,
    type System,
    serialize,
    sparse,
    stringify,
    u32,
} from "../src";
import { clear, entries } from "../src/engine/ecs/core";
import { Compute } from "../src/engine/runtime";
import { LinesPlugin } from "../src/extras/lines";
import { OrbitPlugin } from "../src/extras/orbit";
import { LiveSkin, Skin, SkinPlugin } from "../src/extras/skin";
import { SpritePlugin } from "../src/extras/sprite";
import { TweenPlugin } from "../src/extras/tween";
import { GlazePlugin } from "../src/standard/glaze";
import { InputPlugin } from "../src/standard/input";
import { MirrorPlugin } from "../src/standard/mirror";
import { PartPlugin } from "../src/standard/part";
import { RenderPlugin } from "../src/standard/render";
import { Draws, Surfaces } from "../src/standard/render/core";
import { SearPlugin } from "../src/standard/sear";
import { SlabPlugin } from "../src/standard/slab";
import { TransformsPlugin } from "../src/standard/transforms";

export interface Conformance {
    plugins: Plugin[];
    scene?: string;
    /** snapshot of plugin-owned module registries, compared across builds */
    probe?: () => unknown;
}

// GPU validation errors are reported via the device's uncapturederror → console.error path, invisible
// to test expectations — a rebuild that submits against torn-down resources must fail the sweep, not just
// log. Divert those into `out` for the duration; returns the restore.
export function captureGpuErrors(out: string[]): () => void {
    const origError = console.error;
    console.error = (...args: unknown[]) => {
        if (String(args[0]).includes("GPU uncaptured error")) out.push(args.join(" "));
        else origError(...args);
    };
    return () => {
        console.error = origError;
    };
}

// the rebuild loop: build → step → dispose, twice, against the same module singletons. The second
// build reuses the first's device — the rebuild contract (the editor's ensureDevice; a fresh device
// is the page-reload case, where module scope resets too). Returns the signature divergences
// between the two passes — a compliant plugin returns none.
export async function conform({ plugins, scene, probe }: Conformance): Promise<string[]> {
    clear();
    const passes: Record<string, unknown>[] = [];
    const gpuErrors: string[] = [];
    const restore = captureGpuErrors(gpuErrors);
    let device: GPUDevice | undefined;
    try {
        for (let pass = 0; pass < 2; pass++) {
            const app = await build({ plugins, defaults: false, scene, device });
            device = Compute.device;
            if (app.skipped.length > 0) {
                app.dispose();
                return [`skipped at build (roster entry is missing a dependency): ${app.skipped}`];
            }
            app.state.step();
            app.state.step();
            passes.push(signature(app.state, probe));
            app.dispose();
        }
    } finally {
        restore();
    }
    const violations: string[] = [];
    for (const key of Object.keys(passes[0])) {
        const a = JSON.stringify(passes[0][key]);
        const b = JSON.stringify(passes[1][key]);
        if (a !== b) violations.push(`${key} diverged: first build ${a}, second build ${b}`);
    }
    if (gpuErrors.length > 0) {
        violations.push(`${gpuErrors.length} GPU uncaptured error(s), first: ${gpuErrors[0]}`);
    }
    return violations;
}

// the rebuild loop's toggle variant: build a SEQUENCE of (possibly different) plugin sets in order
// against the same module singletons, clearing the ECS registry before each — mirroring the editor's
// buildState, which clears on every scene-switch / plugin-toggle / play-stop rebuild. Steps sharing a
// plugin set + scene must produce the same observable signature, so toggling a plugin on then back off
// returns the State to its pre-toggle shape. Module-level residue a `clear()` doesn't wipe (the render
// registries are module-level, not ECS-scoped) shows as a divergence between two same-config steps.
export async function conformSequence(steps: Conformance[]): Promise<string[]> {
    const gpuErrors: string[] = [];
    const restore = captureGpuErrors(gpuErrors);
    const violations: string[] = [];
    const byKey = new Map<string, Record<string, unknown>>();
    let device: GPUDevice | undefined;
    try {
        for (let i = 0; i < steps.length; i++) {
            clear();
            const step = steps[i];
            const app = await build({
                plugins: step.plugins,
                defaults: false,
                scene: step.scene,
                device,
            });
            device = Compute.device;
            if (app.skipped.length > 0) {
                app.dispose();
                return [`step ${i} skipped at build (missing a dependency): ${app.skipped}`];
            }
            app.state.step();
            app.state.step();
            const sig = signature(app.state, step.probe);
            app.dispose();
            const key = `${step.plugins
                .map((p) => p.name)
                .sort()
                .join("+")}|${step.scene ?? ""}`;
            const prior = byKey.get(key);
            if (!prior) {
                byKey.set(key, sig);
                continue;
            }
            for (const k of Object.keys(sig)) {
                const a = JSON.stringify(prior[k]);
                const b = JSON.stringify(sig[k]);
                if (a !== b) {
                    violations.push(
                        `step ${i} (${key}): ${k} diverged from an earlier same-config build — ${a} vs ${b}`,
                    );
                }
            }
        }
    } finally {
        restore();
    }
    if (gpuErrors.length > 0) {
        violations.push(`${gpuErrors.length} GPU uncaptured error(s), first: ${gpuErrors[0]}`);
    }
    return violations;
}

// the observable shape a rebuild must reproduce: the registered component set, per-component live
// entity counts (a doubling warm spawn shows here), the serialized document (authored values
// stable), and any plugin-owned registry the entry probes.
export function signature(state: State, probe?: () => unknown): Record<string, unknown> {
    const components: string[] = [];
    const counts: Record<string, number> = {};
    for (const { name, component } of entries()) {
        components.push(name);
        counts[name] = [...state.query([component as never])].length;
    }
    return {
        components: components.sort(),
        counts,
        document: stringify(serialize(state)),
        probe: probe?.(),
    };
}

export const registryNames = (reg: Iterable<{ name: string }>) =>
    [...reg].map((e) => e.name).sort();

// the live-skin substrate's whole reset-scoped layout — the block regions plus what a build published
export const skinLayout = () => ({
    paletteCap: LiveSkin.paletteCap,
    paletteEnd: LiveSkin.paletteEnd,
    jwCap: LiveSkin.jwCap,
    jwEnd: LiveSkin.jwEnd,
    blocks: LiveSkin.blocks.size,
    meshes: LiveSkin.meshes.size,
    params: LiveSkin.params.size,
    published: Compute.buffers.has("skinData"),
});

// a custom project-style plugin — the user-authored shape the editor builds from a manifest's local
// plugins (the hot-reload capture fixture's `ticker`). Project plugins never ran the conformance loop
// before; this pins that a project warm spawns exactly once per build (a doubling warm shows as a
// Counter count of 2 in the second pass) and rebuilds idempotently against the same module singletons.
const Counter = { ticks: sparse(u32) };
const CounterSystem: System = {
    name: "counter",
    group: "simulation",
    annotations: { mode: "always" },
    update: (s) => {
        for (const eid of s.query([Counter])) Counter.ticks.set(eid, Counter.ticks.get(eid) + 1);
    },
};
const ProjectPlugin: Plugin = {
    name: "project-counter",
    components: { Counter },
    systems: [CounterSystem],
    warm: (s) => {
        s.add(s.create(), Counter);
    },
};

// a hand-rig live-skin producer, the shape SkinPlugin exists for: no glTF asset, a mesh's joints/weights
// registered once and a palette block allocated per instance. Derived (warm-spawned), so it never enters
// the serialized document.
const RigPlugin: Plugin = {
    name: "rig",
    dependencies: [SkinPlugin],
    warm: (s) => {
        LiveSkin.registerMesh(0, new Uint32Array(4), new Uint32Array(4));
        const eid = s.create();
        s.add(eid, Skin);
        Skin.anim.x.set(eid, LiveSkin.alloc(eid, 2, s.stamp(eid)));
    },
};

/** the full per-plugin reload-conformance roster — one source of truth, split by `isPipelineCompiling`. */
export const roster: Record<string, Conformance> = {
    "Project plugin": { plugins: [ProjectPlugin] },
    Mirror: { plugins: [MirrorPlugin] },
    Input: { plugins: [InputPlugin] },
    "Slab + Transforms": {
        plugins: [SlabPlugin, TransformsPlugin],
        scene: `<scene><a id="thing" transform="pos: 1 2 3" /></scene>`,
    },
    Orbit: {
        plugins: [SlabPlugin, TransformsPlugin, InputPlugin, OrbitPlugin],
        scene: `<scene><a id="cam" orbit="distance: 8" transform /></scene>`,
    },
    Render: {
        plugins: [SlabPlugin, TransformsPlugin, RenderPlugin],
        scene: `<scene>
            <a ambient-light />
            <a directional-light />
            <a id="cam" camera transform />
        </scene>`,
        probe: () => ({ surfaces: registryNames(Surfaces), draws: registryNames(Draws) }),
    },
    Part: {
        plugins: [SlabPlugin, TransformsPlugin, RenderPlugin, PartPlugin],
        scene: `<scene>
            <a id="cam" camera transform />
            <a id="box" part transform="scale: 2 1 2" color="rgba: 0.8 0.5 0.3 1" />
        </scene>`,
        probe: () => ({ surfaces: registryNames(Surfaces), draws: registryNames(Draws) }),
    },
    Sear: {
        plugins: [SlabPlugin, TransformsPlugin, RenderPlugin, PartPlugin, SearPlugin],
        scene: `<scene>
            <a ambient-light />
            <a directional-light shadow />
            <a id="cam" camera sear transform />
            <a id="box" part transform />
        </scene>`,
        probe: () => ({ surfaces: registryNames(Surfaces), draws: registryNames(Draws) }),
    },
    Glaze: {
        plugins: [SlabPlugin, TransformsPlugin, RenderPlugin, PartPlugin, SearPlugin, GlazePlugin],
        scene: `<scene>
            <a id="cam" camera sear glaze transform />
            <a id="box" part transform />
        </scene>`,
    },
    // Physics / Character / Player can't build on the bun-webgpu adapter (the contact store
    // exceeds its limits — UnsupportedError at build), so they join at the real-GPU tier when
    // a rebuild-loop gym scenario exists, not here.
    Tween: {
        plugins: [SlabPlugin, TransformsPlugin, TweenPlugin],
        scene: `<scene><a id="thing" transform tween="field: transform.pos.y; to: 3; duration: 1" /></scene>`,
    },
    Lines: {
        plugins: [SlabPlugin, TransformsPlugin, RenderPlugin, SearPlugin, LinesPlugin],
        scene: `<scene>
            <a id="cam" camera sear transform />
            <a id="axis" line="offset: 0 5 0" transform />
        </scene>`,
        probe: () => ({ surfaces: registryNames(Surfaces), draws: registryNames(Draws) }),
    },
    Sprite: {
        plugins: [SlabPlugin, TransformsPlugin, RenderPlugin, SearPlugin, SpritePlugin],
        scene: `<scene>
            <a id="cam" camera sear transform />
            <a id="icon" sprite transform />
        </scene>`,
        probe: () => ({ surfaces: registryNames(Surfaces), draws: registryNames(Draws) }),
    },
    // the live joint-palette substrate: a module singleton holding a GPU buffer + a block layout, reset
    // per build. `RigPlugin` is the producer the substrate exists for — it allocates a block + registers
    // a mesh in warm, so the layout the probe reads is non-empty and a reset that stopped clearing would
    // carry the first build's block into the second (appended, not reused) and diverge here.
    Skin: {
        plugins: [SlabPlugin, TransformsPlugin, RenderPlugin, SearPlugin, SkinPlugin, RigPlugin],
        scene: `<scene><a id="cam" camera sear transform /></scene>`,
        probe: () => skinLayout(),
    },
};

/**
 * The declared predicate splitting the roster: true when an entry's plugins include `RenderPlugin`,
 * the plugin whose presence triggers GPU render-pipeline compilation — the cost being promoted to
 * the by-path tier. Entries without it (Project, Mirror, Input, Slab+Transforms, Orbit, Tween) are
 * cheap and stay in the default-tier sentinel.
 */
export const isPipelineCompiling = (entry: Conformance): boolean =>
    entry.plugins.includes(RenderPlugin);
