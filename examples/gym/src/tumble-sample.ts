// The gym tumble-sample host: the one-time layer every ported tumble.js sample runs on. A scenario hands
// this a committed gold trajectory and an escape-hatch `build(world, params)` (the sample's `build()`
// reproduced near-verbatim), and gets, for free:
//   1. the gold-match oracle — a bit-exact ST replay vs the gold, run at build time, reported through the
//      scenario `assert` (the automated correctness gate `shallot verify` / `bun bench` gates on);
//   2. a source-faithful visual layer, derived entirely from the world's own shape data — the `world.draw`
//      walk drives it every frame: solid bodies render as shallot-native instanced Parts (`tumble-solids.ts`,
//      one mesh per unique geometry, colored by the walk's body-state hue), joints + contact points stay
//      lines over the top; visuals can't drift from the verified physics the way hand-built per-scenario
//      meshes did, because everything reads off the stepped world;
//   3. camera framing from the gold's recorded pose, and mouse-grab (left-drag) ported from the `Sample`
//      base — the interaction the first examples port never had.
//
// The host OWNS its `World` (not `TumblePlugin`'s singleton): the sample bodies are authored raw through the
// escape hatch, not marshaled `Body` entities, so nothing but authoring can differ from the gold. One live
// world at a time — the oracle's throwaway world is destroyed before the render world is built (the kernel
// singleton traps on interleaved worlds; run ONE sample scenario per page load, the stage-1 mint finding).

import {
    AmbientLight,
    Camera,
    CameraMode,
    Color,
    DirectionalLight,
    GlazePlugin,
    InputPlugin,
    Inputs,
    Orbit,
    OrbitPick,
    OrbitPlugin,
    Part,
    PartPlugin,
    RenderPlugin,
    run,
    Sear,
    SearPlugin,
    SlabPlugin,
    type State,
    type System,
    Transform,
    TransformsPlugin,
    unpackColor,
} from "@dylanebert/shallot";
import { LinesPlugin, ProfilePlugin, segment } from "@dylanebert/shallot/extras";
import { cursorRay } from "@dylanebert/shallot/physics/core";
import {
    BodyType,
    type DebugDraw,
    defaultDebugDraw,
    World,
    type WorldTransform,
} from "@dylanebert/shallot/tumble/core";
import type { Check, Param, Params, Scenario } from "./gym";
import { beginGrab, driveGrab, endGrab, type Grab, updateGrab } from "./tumble-grab";
import {
    goldParams,
    type OracleResult,
    runOracle,
    type SampleBuild,
    type SampleGold,
    type SampleUpdate,
} from "./tumble-oracle";
import { type OverlayLayer, overlayLayer, type SampleRender } from "./tumble-overlay";
import { type ProbeContext, runProbe } from "./tumble-probe";
import {
    collectSolids,
    meshKey,
    registerSolids,
    type Solids,
    TOTAL_DRAW_BOUNDS,
} from "./tumble-solids";
import { installWatch } from "./tumble-watch";

/** what a gym scenario hands the host: the committed gold + the escape-hatch build that reproduces its
 *  sample. `update` is the sample's per-step hook (a kinematic sweep, a scheduled joint cut) — run before
 *  every `world.step` in both the oracle and the live view, so a sample whose behavior lives in `update()`
 *  stays gold-exact and the two views stay in lockstep. `params` overrides the knob-derived controls (else
 *  the gold's knobs render as rebuild knobs); `name` overrides the scenario name (default the gold slug). */
export interface SampleConfig {
    gold: SampleGold;
    build: SampleBuild;
    update?: SampleUpdate;
    /** the sample's `render(draw)` overlay — its demonstration layer (labels, cast rays, HUD readouts),
     *  drawn against the live world each frame. Visual-only: never runs in the oracle, never mutates. */
    render?: SampleRender;
    params?: Param[];
    name?: string;
}

const FAR = 1000; // pick-ray length for the grab cast

// derive gym controls from the gold's knobs — a rebuild knob per slider/toggle/select (changing one reloads
// and rebuilds the scene from the new value; the oracle always runs at defaults, so a non-default knob is a
// boundedness probe, not a gold check). Buttons are transport-only and dropped.
function knobParams(gold: SampleGold): Param[] {
    const out: Param[] = [];
    for (const k of gold.knobs) {
        if (k.kind === "slider") {
            out.push({ key: k.key, type: "number", default: k.default, rebuild: true });
        } else if (k.kind === "toggle") {
            out.push({ key: k.key, type: "bool", default: k.default, rebuild: true });
        } else if (k.kind === "select") {
            out.push({
                key: k.key,
                type: "select",
                default: k.default,
                options: [k.default],
                rebuild: true,
            });
        }
    }
    return out;
}

/**
 * Build a gym {@link Scenario} that verifies a ported tumble.js sample against its gold and renders it with
 * source-faithful debug draw + mouse-grab. Stage-4 ports call this with the sample's gold + its `build`.
 */
export function sampleScenario(config: SampleConfig): Scenario {
    const { gold } = config;
    let oracle: OracleResult | null = null;
    // the live scene the stage-5 probe drives (pointer drag + visual-presence), set at the end of build.
    let probeCtx: ProbeContext | null = null;
    // the per-sample overlay channel (labels + HUD readout), set in build when the sample has a `render`.
    let overlay: OverlayLayer | null = null;

    return {
        name: config.name ?? gold.slug,
        params: config.params ?? knobParams(gold),

        async build(canvas: HTMLCanvasElement, params: Params) {
            // 1. the oracle — a throwaway world at gold defaults, replayed + hashed, destroyed before the
            // render world exists (one live world per process).
            oracle = await runOracle(gold, config.build, config.update);

            // 2. the render world, then discover its solid geometry from the world's own shape data — the
            // shape count sizes capacity (many bodies, e.g. 725 dominoes) and the meshes register once the
            // device exists. The kernel is already init'd (ST) by the oracle; the world is authored raw
            // through the escape hatch, so nothing but authoring can differ from the gold.
            const world = new World({
                gravity: { x: gold.gravity[0], y: gold.gravity[1], z: gold.gravity[2] },
                enableSleep: gold.enableSleep,
                enableContinuous: gold.enableContinuous,
            });
            const resolved = { ...goldParams(gold), ...params };
            config.build(world, resolved);
            const solids = collectSolids(world);

            const { state, dispose } = await run({
                defaults: false,
                // headroom over the discovered shape count for mid-run spawns (a body recycler, a mover
                // shoving crates) and the light + camera entities.
                capacity: Math.max(1024, solids.instanceCount * 2 + 64),
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
                    LinesPlugin,
                ],
            });

            const keyToMesh = solidMeshIds(solids);

            const amb = state.create();
            state.add(amb, AmbientLight);
            AmbientLight.color.set(amb, 0xd0dcec);
            AmbientLight.intensity.set(amb, 0.7);
            const sun = state.create();
            state.add(sun, DirectionalLight);
            DirectionalLight.color.set(sun, 0xfff4e0);
            DirectionalLight.intensity.set(sun, 1.2);
            DirectionalLight.direction.set(sun, -0.4, -1, -0.55, 0);

            // camera from the gold's recorded orbit pose: a pivot entity at the framed target, the orbit
            // reading yaw/pitch/radius off it. Left orbits, but the grab's pick claims the press over a body
            // (OrbitPick below) so left partitions cleanly: over a body = grab, over nothing = orbit.
            const pivot = state.create();
            state.add(pivot, Transform);
            Transform.pos.set(
                pivot,
                gold.camera.pivot[0],
                gold.camera.pivot[1],
                gold.camera.pivot[2],
                0,
            );

            const cam = state.create();
            state.add(cam, Transform);
            state.add(cam, Camera);
            state.add(cam, Sear);
            state.add(cam, Orbit);
            Camera.mode.set(cam, CameraMode.Perspective);
            Camera.fov.set(cam, (gold.camera.fov * 180) / Math.PI);
            Camera.near.set(cam, gold.camera.near);
            Camera.far.set(cam, gold.camera.far);
            Orbit.target.set(cam, pivot);
            Orbit.yaw.set(cam, gold.camera.yaw);
            Orbit.pitch.set(cam, gold.camera.pitch);
            Orbit.distance.set(cam, gold.camera.radius);
            Orbit.minDistance.set(cam, 0.1);
            Orbit.maxDistance.set(cam, 5000);
            Orbit.minPitch.set(cam, -1.55);
            Orbit.maxPitch.set(cam, 1.55);
            Orbit.panButton.set(cam, 1);

            // contextual left-click: a left press over a dynamic body starts the grab (below), so orbit must
            // stay out of the way there. The claim runs the same pick beginGrab does — a hit dynamic body —
            // so grab and orbit partition the left button exactly (over a body = grab, over nothing = orbit).
            const claim = (): boolean => {
                const ray = cursorRay(state, cam);
                if (!ray) return false;
                const r = world.castRayClosest(
                    { x: ray.origin[0], y: ray.origin[1], z: ray.origin[2] },
                    { x: ray.dir[0] * FAR, y: ray.dir[1] * FAR, z: ray.dir[2] * FAR },
                );
                if (r.hit === false || r.shape === null) return false;
                return r.shape.getBody().getType() === BodyType.Dynamic;
            };
            OrbitPick.claim = claim;
            state.onDispose(() => {
                if (OrbitPick.claim === claim) OrbitPick.claim = undefined;
            });

            let grab: Grab | null = null;
            let prevLeft = false;
            let stepCount = 0;
            const solidLayer = solidPool(state, keyToMesh);
            // the sample's overlay channel (labels over the canvas + HUD lines), only when it has a `render`.
            overlay = config.render ? overlayLayer(cam, canvas) : null;
            const overlayLive = overlay;
            if (overlayLive) state.onDispose(() => overlayLive.dispose());

            // step + grab, at the fixed rate. Order mirrors the sample's Step(): drive the grab anchor, run
            // the sample update, then step (the anchor's target velocity + any update writes are consumed by
            // this step). Grab acquires only on the button's DOWN-edge (`left && !prevLeft`) like the
            // reference's mouseDown — a held-still miss must not re-cast and snag a body that wanders under
            // the cursor.
            const stepper: System = {
                name: `tumble-sample-step:${gold.slug}`,
                group: "fixed",
                update() {
                    const left = Inputs.mouse.left;
                    if (left && !prevLeft && grab === null) {
                        const ray = cursorRay(state, cam);
                        if (ray) {
                            grab = beginGrab(
                                world,
                                { x: ray.origin[0], y: ray.origin[1], z: ray.origin[2] },
                                { x: ray.dir[0] * FAR, y: ray.dir[1] * FAR, z: ray.dir[2] * FAR },
                            );
                        }
                    }
                    if (grab !== null) {
                        if (left) {
                            const ray = cursorRay(state, cam);
                            if (ray) {
                                updateGrab(
                                    grab,
                                    { x: ray.origin[0], y: ray.origin[1], z: ray.origin[2] },
                                    { x: ray.dir[0], y: ray.dir[1], z: ray.dir[2] },
                                );
                            }
                            driveGrab(grab, gold.timeStep);
                        } else {
                            endGrab(grab);
                            grab = null;
                        }
                    }
                    prevLeft = left;
                    config.update?.(world, resolved, gold.timeStep, stepCount);
                    world.step(gold.timeStep, gold.subStepCount);
                    stepCount++;
                },
            };

            // every render frame (simulation group, before the pack + Lines flush in the draw group): the
            // one walk drives both layers — solid shapes reconcile onto Part instances, joints + points
            // stay lines. `begin`/`end` bracket the walk so a body destroyed or slept-out-of-bounds this
            // frame releases its Part slot (membership follows the world's live shape set).
            const draw: System = {
                name: `tumble-sample-draw:${gold.slug}`,
                group: "simulation",
                update() {
                    solidLayer.begin();
                    world.draw(solidLayer.adapter);
                    solidLayer.end();
                    // the per-sample overlay: its `render()` draws against the live world, after the solids
                    // walk so its gizmo lines sit over them. Visual-only — reads the world, never mutates it.
                    if (overlayLive && config.render) {
                        overlayLive.begin();
                        config.render(overlayLive.api, world, resolved, stepCount);
                        overlayLive.end();
                    }
                },
            };

            state.addSystem(stepper);
            state.addSystem(draw);

            probeCtx = {
                state,
                world,
                cam,
                canvas,
                gold,
                overlay,
                grab: () => {
                    if (grab === null) return null;
                    const a = grab.anchor.getPosition();
                    const b = grab.body.getPosition();
                    return {
                        anchor: [a.x, a.y, a.z],
                        target: [grab.target.x, grab.target.y, grab.target.z],
                        body: [b.x, b.y, b.z],
                        depth: grab.depth,
                    };
                },
            };
            // the floor-vanish instrument (spec 6b/F1): passive page hooks always, the auto-dump watcher
            // under `watch=1`. Diagnostic-only — reads world/Part state, never feeds the oracle.
            installWatch(probeCtx);

            return {
                state,
                dispose() {
                    if (grab !== null) {
                        endGrab(grab);
                        grab = null;
                    }
                    probeCtx = null;
                    overlay = null;
                    world.destroy();
                    dispose();
                },
            };
        },

        async assert(): Promise<Check[]> {
            if (oracle === null)
                return [{ name: "gold", pass: false, detail: "oracle did not run" }];
            if (oracle.pass) {
                return [
                    {
                        name: "gold",
                        pass: true,
                        detail: `bit-exact vs ${gold.slug} for ${oracle.steps} steps`,
                        data: { steps: oracle.steps },
                    },
                ];
            }
            return [
                {
                    name: "gold",
                    pass: false,
                    detail: `diverged at step ${oracle.step}: got ${oracle.got}, expected ${oracle.expected}`,
                    data: { firstDivergentStep: oracle.step },
                },
            ];
        },

        async probe(_state, opts): Promise<Check[]> {
            if (!probeCtx) return [];
            return runProbe(probeCtx, opts);
        },

        live(): string {
            if (oracle === null) return `${gold.slug} — building`;
            const status = oracle.pass
                ? `${gold.slug} — gold OK (${oracle.steps} steps)`
                : `${gold.slug} — DIVERGED @ step ${oracle.step}`;
            // the sample's `drawText` readout, below the gold status (empty for a sample with no overlay).
            const lines = overlay?.hudLines() ?? [];
            return lines.length ? `${status}\n${lines.join("\n")}` : status;
        },
    };
}

// register the discovered geometry (device now exists) and resolve each shape's stable geometry key to its
// mesh id, so the per-frame reconcile sets `Part.mesh` without a name lookup.
function solidMeshIds(solids: Solids): Map<object, number> {
    const nameToId = registerSolids(solids.defs);
    const out = new Map<object, number>();
    for (const [key, name] of solids.keyToName) {
        const id = nameToId.get(name);
        if (id !== undefined) out.set(key, id);
    }
    return out;
}

// The solid layer: the walk's per-shape solid callbacks reconcile onto pooled Part instances, keyed by mesh.
// Instances of one mesh are fungible (identical geometry, per-frame color + transform), so the k-th shape of
// a mesh this frame drives the k-th pooled entity — a stable count, not a stable body→entity binding. The
// draw order churns frame to frame, but the rendered set of (transform, color) is exactly the walk's output.
// `end` releases (drops `Part`) the surplus a shrunk count leaves behind, so a destroyed / culled body's slot
// stops rendering the next frame; `begin` re-adds it when the count grows back.
export function solidPool(
    state: State,
    keyToMesh: Map<object, number>,
): { adapter: DebugDraw; begin(): void; end(): void } {
    const pools = new Map<number, number[]>();
    const active = new Map<number, number>();
    const cursor = new Map<number, number>();

    const place = (meshId: number, xf: WorldTransform, color: number): void => {
        let pool = pools.get(meshId);
        if (!pool) {
            pool = [];
            pools.set(meshId, pool);
        }
        const i = cursor.get(meshId) ?? 0;
        let eid: number;
        if (i < pool.length) {
            eid = pool[i];
            if (i >= (active.get(meshId) ?? 0)) state.add(eid, Part);
        } else {
            eid = state.create();
            state.add(eid, Transform);
            state.add(eid, Color);
            state.add(eid, Part);
            pool.push(eid);
        }
        Part.mesh.set(eid, meshId);
        const q = xf.q;
        Transform.pos.set(eid, xf.p.x, xf.p.y, xf.p.z, 0);
        Transform.rot.set(eid, q.v.x, q.v.y, q.v.z, q.s);
        Transform.scale.set(eid, 1, 1, 1, 1);
        const c = unpackColor(color);
        Color.rgba.set(eid, c.r, c.g, c.b, 1);
        cursor.set(meshId, i + 1);
    };

    const solid = (key: object, xf: WorldTransform, color: number): void => {
        const meshId = keyToMesh.get(key);
        if (meshId !== undefined) place(meshId, xf, color);
    };

    const a = [0, 0, 0];
    const adapter: DebugDraw = {
        ...defaultDebugDraw(),
        // total derivation: every world solid body reconciles to a Part regardless of position — the Part
        // pack's frustum cull owns on-screen visibility, not this walk (`tumble-solids.ts` TOTAL_DRAW_BOUNDS).
        drawingBounds: TOTAL_DRAW_BOUNDS,
        drawShapes: true,
        drawJoints: true,
        drawSolidSphere: (xf, sphere, color) => solid(sphere, xf, color),
        drawSolidCapsule: (xf, cap, color) => solid(cap, xf, color),
        drawSolidHull: (xf, hull, color) => solid(hull, xf, color),
        drawSolidMesh: (xf, mesh, color) => solid(meshKey(mesh), xf, color),
        // joints + points stay lines, drawn over the solids each frame.
        drawSegment(p1, p2, color) {
            segment([p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z], color);
        },
        drawPoint(p, size, color) {
            const r = 0.02 * Math.max(size, 1);
            segment([p.x - r, p.y, p.z], [p.x + r, p.y, p.z], color, 3);
            segment([p.x, p.y - r, p.z], [p.x, p.y + r, p.z], color, 3);
            segment([p.x, p.y, p.z - r], [p.x, p.y, p.z + r], color, 3);
        },
        drawTransform(xf) {
            const o = [xf.p.x, xf.p.y, xf.p.z];
            rotate(xf, 0.5, 0, 0, a);
            segment(o, a, 0xff0000);
            rotate(xf, 0, 0.5, 0, a);
            segment(o, a, 0x00ff00);
            rotate(xf, 0, 0, 0.5, a);
            segment(o, a, 0x0000ff);
        },
    };

    return {
        adapter,
        begin() {
            cursor.clear();
        },
        end() {
            for (const [meshId, pool] of pools) {
                const count = cursor.get(meshId) ?? 0;
                const prev = active.get(meshId) ?? 0;
                for (let i = count; i < prev; i++) state.remove(pool[i], Part);
                active.set(meshId, count);
            }
        },
    };
}

// local +axis (0.5 m) → world, for the joint-frame RGB gizmo (drawTransform).
function rotate(xf: WorldTransform, lx: number, ly: number, lz: number, out: number[]): void {
    const q = xf.q;
    const vx = q.v.x;
    const vy = q.v.y;
    const vz = q.v.z;
    const w = q.s;
    const tx = 2 * (vy * lz - vz * ly);
    const ty = 2 * (vz * lx - vx * lz);
    const tz = 2 * (vx * ly - vy * lx);
    out[0] = xf.p.x + lx + w * tx + (vy * tz - vz * ty);
    out[1] = xf.p.y + ly + w * ty + (vz * tx - vx * tz);
    out[2] = xf.p.z + lz + w * tz + (vx * ty - vy * tx);
}
