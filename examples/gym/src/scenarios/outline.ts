import {
    AmbientLight,
    Camera,
    CameraMode,
    Color,
    Depth,
    DirectionalLight,
    Fog,
    FogPlugin,
    GlazePlugin,
    InputPlugin,
    Orbit,
    OrbitPlugin,
    Part,
    PartPlugin,
    type Plugin,
    RenderPlugin,
    run,
    Sear,
    SearPlugin,
    SlabPlugin,
    type State,
    Transform,
    TransformsPlugin,
} from "@dylanebert/shallot";
import { Outline, OutlinePlugin, ProfilePlugin } from "@dylanebert/shallot/extras";
import { type Check, frames, type Params, register, type Scenario } from "../gym";

// outline — the screen-space OutlinePlugin atom. A spread of cube Parts: two highlighted always-on-top
// (a wide yellow band + a thin white one, proving per-entity color/width coexist in one pass), one behind
// a wall highlighted occlusion-aware (its band hides where the wall hides it — the camera carries sear's
// `Depth` marker so `view.depth` exists), and one at the right screen edge past x 1024 (the seed-precision
// visual gate — integer seed coords; an f16 seed regression washes its interior and breaks its band).
// Drive the orbit camera live to read the band at any angle.
//
// `fog` adds FogPlugin in the post-color seam — the interaction gate this `outline` atom's `fog` mode holds.
// FogPlugin (a scene-transform compute effect, repoints `view.framebuffer` at the rgba16float scratch) and
// OutlinePlugin (a screen-space overlay) both run in the seam; the `OverlaySystem` anchor must order fog
// before the outline composite regardless of registration order, so the band lands on top of the haze with
// no validation error either way (the harness fails the run on any WebGPU validation error). `fogFirst`
// flips the plugin order so both former-failure orders boot clean.
//
// The bands' shape + the occlusion clip + the on-top-of-fog look are visual — read them in the render. The
// headless assert can't pixel-read (the offscreen `view.framebuffer` isn't surfaced once the harness stops
// presenting), so it gates what the profiler sees: the `outline:*` passes run while something is
// highlighted and none when nothing is; with fog, both effects composite through the seam.

let highlightedA = -1; // wide yellow, always-on-top
let highlightedB = -1; // thin white, always-on-top
let occluded = -1; //     magenta, occlusion-aware, behind the wall
let edge = -1; //      orange, always-on-top, at screen x > 1024 (pins integer seed coords — a f16
//                       seed regression washes its interior and breaks its band; see SEED_FORMAT)
let occludeRear = true; // the `occlude` param value — the assert's restore reads it so the screenshot honors it

function box(
    state: State,
    pos: [number, number, number],
    scale: [number, number, number],
    color: [number, number, number],
): number {
    const eid = state.create();
    state.add(eid, Transform);
    Transform.pos.set(eid, pos[0], pos[1], pos[2], 0);
    Transform.scale.set(eid, scale[0], scale[1], scale[2], 0);
    state.add(eid, Part);
    state.add(eid, Color);
    Color.rgba.set(eid, color[0], color[1], color[2], 1);
    return eid;
}

// the four highlights, applied in build and re-applied after the assert removes them (so the post-run
// screenshot + live view show the bands): a wide warm-yellow always-on-top, a thin white always-on-top,
// a magenta band behind the wall whose `occlude` flag is the depth-test switch, and the precision-edge box.
function highlight(state: State, occludeRear: boolean): void {
    state.add(highlightedA, Outline);
    Outline.color.set(highlightedA, 1, 0.85, 0.2, 1);
    Outline.width.set(highlightedA, 8);
    state.add(highlightedB, Outline);
    Outline.color.set(highlightedB, 1, 1, 1, 1);
    Outline.width.set(highlightedB, 3);
    state.add(occluded, Outline);
    Outline.color.set(occluded, 1, 0.2, 0.7, 1);
    Outline.width.set(occluded, 5);
    Outline.occlude.set(occluded, occludeRear ? 1 : 0);
    state.add(edge, Outline);
    Outline.color.set(edge, 0.9, 0.6, 0.2, 1);
    Outline.width.set(edge, 3);
}

const hasPass = (passes: Record<string, unknown>, prefix: string): boolean =>
    Object.keys(passes).some((k) => k.startsWith(prefix));

const scenario: Scenario = {
    name: "outline",
    params: [
        { key: "occlude", type: "bool", default: true, label: "occlude rear box" },
        { key: "fog", type: "bool", default: false, rebuild: true, label: "fog seam" },
        // flip the plugin registration order — the anchor must order fog before the outline composite
        // either way (with the old undefined ordering, one order raised a validation error and the other
        // fogged the band)
        {
            key: "fogFirst",
            type: "bool",
            default: false,
            rebuild: true,
            when: (v) => v.fog === true,
        },
    ],

    async build(_canvas: HTMLCanvasElement, p: Params) {
        const seam: Plugin[] = p.fog
            ? p.fogFirst
                ? [FogPlugin, OutlinePlugin]
                : [OutlinePlugin, FogPlugin]
            : [OutlinePlugin];
        const { state, dispose } = await run({
            defaults: false,
            capacity: 64,
            plugins: [
                ProfilePlugin,
                SlabPlugin,
                TransformsPlugin,
                InputPlugin,
                OrbitPlugin,
                RenderPlugin,
                PartPlugin,
                SearPlugin,
                GlazePlugin,
                ...seam,
            ],
        });

        state.add(state.create(), AmbientLight);
        const sun = state.create();
        state.add(sun, DirectionalLight);
        DirectionalLight.direction.set(sun, -0.4, -0.8, -0.45, 0);

        // a ground slab + two side cubes (always-on-top, off to the sides so the centre stays clear)
        box(state, [0, -1.2, 0], [10, 0.4, 10], [0.26, 0.28, 0.3]);
        highlightedA = box(state, [-3.4, 0, 0], [1, 1, 1], [0.7, 0.5, 0.35]);
        highlightedB = box(state, [3.4, 0, 0], [1, 1, 1], [0.4, 0.55, 0.7]);

        // centre: a tall box with a wide opaque wall in front of it (toward the camera at +z; the orbit eye
        // sits at +z). The wall is wider + taller than the box's lower half and sits in the Depth prepass, so
        // it covers the box's lower silhouette — the occlusion-aware band hides there (`occlude` on) and the
        // box top pokes above it with its band intact. `occlude` off draws the full band over the wall.
        occluded = box(state, [0, 0.2, 0], [1.2, 1.8, 1.2], [0.6, 0.4, 0.55]);
        box(state, [0, -0.7, 1.0], [3.2, 1.4, 0.3], [0.32, 0.34, 0.4]);

        // right edge: a box past screen x 1024 in the start pose, where f16 seed coords lose the
        // pixel fraction — its band must match the left boxes' (the seed-precision visual gate)
        edge = box(state, [6.2, 0.6, 2], [1, 1, 1], [0.55, 0.38, 0.25]);

        occludeRear = p.occlude as boolean;
        highlight(state, occludeRear);

        if (p.fog) {
            const fog = state.create();
            state.add(fog, Fog);
            Fog.density.set(fog, 0.04);
            Fog.color.set(fog, 0xb5c4d8);
            Fog.heightFalloff.set(fog, 0.1);
        }

        const cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Camera);
        state.add(cam, Sear);
        state.add(cam, Depth); // sear's depth lane → view.depth, for the occlusion-aware band + the fog march
        state.add(cam, Orbit);
        Camera.mode.set(cam, CameraMode.Perspective);
        Camera.fov.set(cam, 55);
        Orbit.distance.set(cam, 10);
        Orbit.pitch.set(cam, Math.PI / 12);
        Orbit.yaw.set(cam, 0);

        return {
            state,
            dispose() {
                highlightedA = -1;
                highlightedB = -1;
                occluded = -1;
                edge = -1;
                dispose();
            },
        };
    },

    // the zero-cost gate: the outline runs the `outline:*` GPU passes while the boxes are highlighted, and
    // none after every `Outline` is removed — the bare-path early-out holds on the real GPU. With fog, the
    // seam gate: fog:march + outline:composite both run in one frame (the order anchor's clean-boot proof,
    // backed by the harness's validation-error fail). The band shape + occlusion clip stay visual.
    async assert(state: State): Promise<Check[]> {
        const checks: Check[] = [];
        const bench = window.__benchmark;
        if (!bench) return checks;
        const fog = [...state.query([Fog])].length > 0;

        await frames(3);
        const lit = (await bench.measure(2, 20)).gpu?.passes ?? {};
        checks.push({
            name: "outline: highlighted entities run the mask → JFA → composite passes",
            pass: hasPass(lit, "outline:"),
            detail: `gpu passes: ${Object.keys(lit).join(", ") || "(none)"}`,
        });
        if (fog) {
            checks.push({
                name: "outline + fog both composite through the seam",
                pass: hasPass(lit, "fog:march") && hasPass(lit, "outline:composite"),
                detail: `gpu passes: ${Object.keys(lit).join(", ") || "(none)"}`,
            });
        }

        for (const eid of [...state.query([Outline])]) state.remove(eid, Outline);
        await frames(3);
        const bare = (await bench.measure(2, 20)).gpu?.passes ?? {};
        checks.push({
            name: "outline: nothing highlighted → zero outline passes",
            pass: !hasPass(bare, "outline:"),
            detail: `gpu passes: ${Object.keys(bare).join(", ") || "(none)"}`,
        });

        // restore the highlights so the post-run screenshot + live view show the bands (honoring `occlude`)
        highlight(state, occludeRear);
        await frames(2);
        return checks;
    },

    live(): string {
        return "outline — drag to orbit; yellow/white always-on-top, magenta occluded by the wall, orange at the precision edge";
    },
};

register(scenario);
