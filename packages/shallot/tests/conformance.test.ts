// the per-plugin reload-conformance sweep (testing.md "Reload tier"): a live host that rebuilds a
// State (a rejected swap's fallback, a restore) re-runs every plugin's lifecycle against the SAME
// module-level singletons (ecs.md "Reload-safety"). The mechanism tests pin the core seams (ids,
// swap, rebuild, serialize); this harness pins each plugin's module-scope state — two identical
// build→step→dispose passes must produce the same observable State, so a registry that
// double-registers or a warm that doubles its derived spawns goes red here. The browser end-to-end
// is the survive-reload flow at examples/flows/survive-reload/ (`bun run flows`), which rebuilds through a real page
// reload; the editor's live rebuild-loop e2e died with the editor, so this roster is the sole
// per-plugin conformance coverage.

import { describe, expect, test } from "bun:test";
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

interface Conformance {
    plugins: Plugin[];
    scene?: string;
    /** snapshot of plugin-owned module registries, compared across builds */
    probe?: () => unknown;
}

// GPU validation errors are reported via the device's uncapturederror → console.error path, invisible
// to test expectations — a rebuild that submits against torn-down resources must fail the sweep, not just
// log. Divert those into `out` for the duration; returns the restore.
function captureGpuErrors(out: string[]): () => void {
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
async function conform({ plugins, scene, probe }: Conformance): Promise<string[]> {
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
async function conformSequence(steps: Conformance[]): Promise<string[]> {
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
function signature(state: State, probe?: () => unknown): Record<string, unknown> {
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

const registryNames = (reg: Iterable<{ name: string }>) => [...reg].map((e) => e.name).sort();

describe("reload conformance", () => {
    // the harness itself catches a violation: a module-level registry initialize fails to clear,
    // so the second build's warm derives one entity per accumulated entry — the doubling shape
    test("a seeded non-idempotent plugin goes red", async () => {
        const ledger: number[] = []; // module-scope registry, never cleared — the violation
        const Bad = { n: sparse(u32) };
        const BadPlugin: Plugin = {
            name: "bad",
            components: { Bad },
            initialize: () => {
                ledger.push(1);
            },
            warm: (s) => {
                for (const _ of ledger) s.add(s.create(), Bad);
            },
        };
        const violations = await conform({ plugins: [BadPlugin] });
        expect(violations.length).toBeGreaterThan(0);
        expect(violations.join("\n")).toContain("counts");
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
            for (const eid of s.query([Counter]))
                Counter.ticks.set(eid, Counter.ticks.get(eid) + 1);
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

    const roster: Record<string, Conformance> = {
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
            plugins: [
                SlabPlugin,
                TransformsPlugin,
                RenderPlugin,
                PartPlugin,
                SearPlugin,
                GlazePlugin,
            ],
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
    };

    for (const [name, entry] of Object.entries(roster)) {
        test(`${name} rebuilds idempotently`, async () => {
            expect(await conform(entry)).toEqual([]);
        });
    }

    // the plugin-toggle rebuild path: a rebuild with a CHANGED plugin set (the editor's toggle path).
    // Toggling an extra on then back off must return the State to its pre-toggle shape — no residue from
    // the toggled-out plugin. A non-producer (Tween: a component + a system, no GPU registries) isolates
    // the plain toggle from the producer-registry case below.
    test("a non-producer plugin toggle leaves no residue", async () => {
        const base: Plugin[] = [SlabPlugin, TransformsPlugin];
        const scene = `<scene><a id="thing" transform="pos: 1 2 3" /></scene>`;
        const violations = await conformSequence([
            { plugins: base, scene },
            { plugins: [...base, TweenPlugin], scene },
            { plugins: base, scene },
        ]);
        expect(violations).toEqual([]);
    });

    // a producer toggle is the strict case: Lines registers a surface + a draw into the module-level
    // render registries (not ECS-scoped, so `clear()` doesn't touch them). Toggling it off must leave the
    // surface / draw set as it was before — a stale draw against torn-down buffers would show as a Surfaces/
    // Draws divergence or a GPU uncaptured error on the post-toggle build.
    test("a producer plugin toggle leaves no residue", async () => {
        const base: Plugin[] = [SlabPlugin, TransformsPlugin, RenderPlugin, SearPlugin, PartPlugin];
        const boxScene = `<scene>
            <a id="cam" camera sear transform />
            <a id="box" part transform />
        </scene>`;
        const lineScene = `<scene>
            <a id="cam" camera sear transform />
            <a id="box" part transform />
            <a id="axis" line="offset: 0 5 0" transform />
        </scene>`;
        const probe = () => ({ surfaces: registryNames(Surfaces), draws: registryNames(Draws) });
        const violations = await conformSequence([
            { plugins: base, scene: boxScene, probe },
            { plugins: [...base, LinesPlugin], scene: lineScene, probe },
            { plugins: base, scene: boxScene, probe },
        ]);
        expect(violations).toEqual([]);
    });
});
