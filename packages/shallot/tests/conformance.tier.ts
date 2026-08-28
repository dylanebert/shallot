// the by-path tier holding the GPU pipeline-compiling arms promoted out of `conformance.test.ts`.
//
// Reason: real GPU pipeline-compile cost. The promoted arms (Render, Part, Sear, Glaze, Lines,
// Sprite, Skin, Physics, Character, Player roster entries; both producer/non-producer toggle arms;
// the SkinPlugin+GltfPlugin pair; the shared-trait ordering arm) each build through `RenderPlugin`,
// which compiles GPU render pipelines — or, for the physics stack (Physics, Character, Player),
// through `AvbdPlugin`, whose build compiles the AVBD solver's compute pipeline set on the device at
// a threaded entity capacity (8192 — the contact store fits under the bun-webgpu adapter's 128 MiB
// binding ceiling; the headless build/step/readback measurement lives in
// `tests/avbd/headless.tier.ts`'s header). Measured 2026-08-26 on trunk: the full 18-arm file ran 4.6–6.3 s, straddling the
// 5000 ms per-file cap (`tests/test-cap.ts`) — readings 5062–6279 ms across isolated runs, so the
// verdict was a host-load coin flip (4-in-6 red), and the blamed arm was whichever test crossed the
// budget. No conformance assertion ever failed; every red was `per-file test cap exceeded`. The
// split is the cap's own prescribed move (2) — `test-cap.ts`'s message names promotion to a by-path
// tier as one of exactly two responses to an armed red; move (1) (derive the scan's N from the
// property) does not apply because the roster's size IS its property (one arm per plugin, each
// pipeline compile is the cost being verified as idempotent).
//
// Destination suffix: `.tier.ts` — confirmed against `testing.md`'s tier-section bullet: "`.tier.ts`
// — a corpus-scale gate split out of the default suite for speed; coverage moved tiers, not shrunk
// — a sentinel stays behind in the default suite against the same frozen fixture." `.oracle.ts` is the
// heavy deterministic CPU reference tier (the f64 AVBD physics oracle), not this case.
//
// Trigger cone (derived per `checks.md`'s by-path rule, `coding.md` Suite speed): this file's own
// transitive import cone — the plugin source modules whose reload conformance the promoted arms pin,
// walked from this file's import statements. The cone is the source tree of every plugin this file
// (and the shared roster it imports) pulls in for a promoted arm: `src/standard/render/`,
// `src/standard/part/`, `src/standard/sear/`, `src/standard/glaze/`, `src/extras/lines/`,
// `src/extras/sprite/`, `src/extras/skin/`, `src/extras/gltf/`, `src/extras/tween/`, plus the
// `src/standard/slab/` and `src/standard/transforms/` trees every promoted arm builds on
// (`SlabPlugin`/`TransformsPlugin` ride in as `RenderPlugin`'s declared `dependencies`), plus —
// for the physics-stack arms — `src/standard/avbd/` (the solver), `src/standard/physics/`
// (the substrate), `src/standard/character/` + `src/standard/player/` (the sweep + controller),
// and `src/standard/mirror/` + `src/standard/input/` (their readback + input dependencies).
// Re-derive this list from the imports above if it drifts — the derivation is the operative rule,
// not the enumeration. A by-path tier file's own header is its registry (`test-cap.ts:96`).
//
// Run by path from the shallot root: `bun test ./packages/shallot/tests/conformance.tier.ts`.
//
// GPUDevice observation: `conform()` threads one `GPUDevice` forward across its two passes via
// `Compute.device` (the module-level singleton). The second pass reuses the first's device by design
// (the rebuild contract). The spec's speculation — a forward-threaded device makes later roster arms
// progressively slower — is refuted: timing across isolated runs shows no monotonic increase (e.g.
// Sear 376–821 ms, Skin 154–320 ms, with later arms sometimes faster), so the singleton does not
// accumulate measurable cost across arms.
// Each `conform()` call starts with `device = undefined` and acquires a fresh device via `requestGPU`,
// so cross-arm accumulation is not the mechanism. A device-lifecycle fix is out of scope for this
// stage.

import { describe, expect, test } from "bun:test";
import { build, type Plugin } from "../src";
import { clear, getComponent, getTraits } from "../src/engine/ecs/core";
import { GltfPlugin } from "../src/extras/gltf";
import { LinesPlugin } from "../src/extras/lines";
import { LiveSkinSystem, Skin, SkinPlugin, skinTraits } from "../src/extras/skin";
import { TweenPlugin } from "../src/extras/tween";
import { PartPlugin } from "../src/standard/part";
import { RenderPlugin } from "../src/standard/render";
import { Draws, Surfaces } from "../src/standard/render/core";
import { SearPlugin } from "../src/standard/sear";
import { SlabPlugin } from "../src/standard/slab";
import { TransformsPlugin } from "../src/standard/transforms";
import {
    conform,
    conformSequence,
    isPipelineCompiling,
    registryNames,
    roster,
    skinLayout,
} from "./conformance-roster";

describe("reload conformance (pipeline-compiling tier)", () => {
    for (const [name, entry] of Object.entries(roster)) {
        if (!isPipelineCompiling(entry)) continue;
        test(`${name} rebuilds idempotently`, async () => {
            expect(await conform(entry)).toEqual([]);
        });
    }

    // the plugin-toggle rebuild path: a rebuild with a CHANGED plugin set.
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

    // SkinPlugin and GltfPlugin both wire the live-skin substrate — the same `Skin` component, the same
    // `LiveSkinSystem`, the same reset/dispose — because GltfPlugin can't take the substrate as a hard
    // dependency without breaking every existing glTF app's plugin list. `SkinPlugin`'s JSDoc claims having
    // both is harmless; these pin it. Component registration is last-writer-wins on the traits, so the
    // order-sweep is what proves the two plugins agree on one traits object rather than racing.
    const skinBase: Plugin[] = [SlabPlugin, TransformsPlugin, RenderPlugin, SearPlugin];
    const skinScene = `<scene><a id="cam" camera sear transform /></scene>`;

    test("SkinPlugin + GltfPlugin in one app rebuilds idempotently", async () => {
        const violations = await conform({
            plugins: [...skinBase, SkinPlugin, GltfPlugin],
            scene: skinScene,
            probe: () => ({ ...skinLayout(), surfaces: registryNames(Surfaces) }),
        });
        expect(violations).toEqual([]);
    });

    test("either plugin order registers the one Skin component on the shared traits", async () => {
        for (const order of [
            [...skinBase, SkinPlugin, GltfPlugin],
            [...skinBase, GltfPlugin, SkinPlugin],
        ]) {
            clear();
            const app = await build({ plugins: order, defaults: false, scene: skinScene });
            expect(app.skipped).toEqual([]);
            app.state.step();
            expect(getComponent("skin")).toBe(Skin);
            expect(getTraits("skin")).toBe(skinTraits);
            // the system is registered by identity, so the second plugin's copy is not a second slot
            expect(app.state.hasSystem(LiveSkinSystem)).toBe(true);
            app.dispose();
        }
    });
});
