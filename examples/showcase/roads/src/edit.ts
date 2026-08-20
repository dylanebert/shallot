// Stage 4 — handles and drag (`roads-interactive.md` stage 4). Two Part `sphere` entities at the
// endpoints, a `Color` slab carrying idle / hover / grabbed, hover via `cursorRay` → `raySphere`, and
// `OrbitPick.claim` so a press over a handle suppresses orbit rotation for the whole drag. On a claimed
// press, each frame marches `flattenFieldAt` along the cursor ray to find the world point, clamps to
// the world bounds, and clamps to the `ROAD_MIN_LENGTH` floor (never refuses — a constraint on a dragged
// quantity clamps, never no-ops). When the march returns null (the ray misses the surface), the drag
// holds its last valid target instead of skipping the frame. The pure, device-free halves (`applyEdit`,
// `clampToBound`, `clampDragTarget`, `chordLength`, `residentTileCount`, `HANDLE_RADIUS`) live in
// `editPure.ts`, which imports nothing from `@dylanebert/shallot`; this module imports them from there
// and re-exports them for consumers like `boot.ts`. `edit.test.ts` imports the pure halves from
// `./editPure` so it exercises them under `bun test` without pulling in the engine's device-bound module
// graph; the Playwright Node side stays bridge-only for the same reason (Node ≥26 rejects the package's
// bare `package.json` import).

// re-export the pure halves for consumers that imported them from ./edit (e.g. boot.ts)
export {
    applyEdit,
    chordLength,
    clampDragTarget,
    clampToBound,
    HANDLE_RADIUS,
    residentTileCount,
} from "./editPure";

// --- the grab latch (pure, device-free state machine) ---
//
// `stepGrab` is the press-edge latch the spec's stage 4b names: on the rising edge of `left` while
// `hovered >= 0`, record the handle index and set `dragging`; `dragging` survives every subsequent
// frame regardless of hover and clears only on the falling edge; a press with no handle under it
// latches nothing. Extracted as a pure function so `edit.test.ts` can drive a synthetic press →
// move-off → move-back → release sequence without a device.

interface GrabState {
    dragging: boolean;
    dragEnd: 0 | 1;
    prevLeft: boolean;
}

export function createGrabState(): GrabState {
    return { dragging: false, dragEnd: 0, prevLeft: false };
}

export function stepGrab(state: GrabState, left: boolean, hovered: number): GrabState {
    const rising = left && !state.prevLeft;
    let dragging = state.dragging;
    let dragEnd = state.dragEnd;
    if (rising && hovered >= 0) {
        dragging = true;
        dragEnd = hovered as 0 | 1;
    }
    if (!left) {
        dragging = false;
    }
    return { dragging, dragEnd, prevLeft: left };
}

import { applyEdit, clampDragTarget, clampToBound, HANDLE_RADIUS } from "./editPure";

// --- the device-bound plugin (imports from @dylanebert/shallot below this line) ---

import {
    Camera,
    Color,
    Inputs,
    Part,
    type Plugin,
    type State,
    type System,
    Transform,
} from "@dylanebert/shallot";
import { OrbitPick } from "@dylanebert/shallot/extras";
import { cursorRay, type Ray, raySphere } from "@dylanebert/shallot/physics/core";
import { Meshes, Surfaces } from "@dylanebert/shallot/render/core";
import { flattenFieldAt } from "./flatness";
import { buildNetworkGeometry, computeFalloff, type ProfileSegment } from "./terrain/flatten";
import { makePermutation } from "./terrain/noise";
import { heightAtCpu } from "./terrain/profile";
import { editDocument, getDocument, SEED } from "./terrain/terrain";

// handle colours — idle (white), hover (yellow), grabbed (orange). Design parameters, not gates: the
// spec's drag-feel check-in is about lag/jump/detach, not colour choice.
const IDLE_COLOR: readonly [number, number, number] = [0.85, 0.85, 0.85];
const HOVER_COLOR: readonly [number, number, number] = [0.9, 0.8, 0.2];
const GRABBED_COLOR: readonly [number, number, number] = [0.95, 0.55, 0.1];

// the cursor-ray march: step size and max distance. 1 m steps are finer than the grid's 4 m spacing, so
// the march catches the surface within one cell; 2000 m reaches past the orbit's max distance.
const MARCH_STEP = 1;
const MARCH_MAX = 2000;

let liveState: State | null = null;
let camEid = -1;
let handleEids: [number, number] = [-1, -1];
let hovered = -1;
let grab = createGrabState();
let claimInstalled = false;
let lastValidTarget: [number, number] | null = null;

/** march `ray` against the continuous flattened field (`flattenFieldAt`) and return the (x, z) where the
 *  ray crosses the surface, or null if it doesn't within `MARCH_MAX`. The field is the oracle's own
 *  mirror — CPU, no GPU readback on the interaction path (the spec's Locked decision). */
function marchFlattenField(
    ray: Ray,
    segments: readonly ProfileSegment[],
    falloff: number,
    naturalHeightAt: (x: number, z: number) => number,
): [number, number] | null {
    const [ox, oy, oz] = ray.origin;
    const [dx, dy, dz] = ray.dir;
    let prevDelta = oy - flattenFieldAt(ox, oz, segments, falloff, naturalHeightAt);
    for (let t = MARCH_STEP; t <= MARCH_MAX; t += MARCH_STEP) {
        const x = ox + dx * t;
        const y = oy + dy * t;
        const z = oz + dz * t;
        const fieldY = flattenFieldAt(x, z, segments, falloff, naturalHeightAt);
        const delta = y - fieldY;
        if (prevDelta > 0 !== delta > 0) return [x, z];
        prevDelta = delta;
    }
    return null;
}

/** create the two Part `sphere` handle entities at the document's endpoints, with `Color` set to idle. */
function createHandles(state: State): [number, number] {
    const sphereMesh = Meshes.id("sphere");
    const unlitSurface = Surfaces.id("unlit");
    const scale = HANDLE_RADIUS / 0.5; // sphere mesh radius 0.5 → world radius HANDLE_RADIUS
    const eids: [number, number] = [-1, -1];
    for (let i = 0; i < 2; i++) {
        const eid = state.create();
        state.add(eid, Transform);
        Transform.scale.set(eid, scale, scale, scale, 0);
        state.add(eid, Part);
        if (sphereMesh !== undefined) Part.mesh.set(eid, sphereMesh);
        if (unlitSurface !== undefined) Part.surface.set(eid, unlitSurface);
        state.add(eid, Color);
        Color.rgba.set(eid, IDLE_COLOR[0], IDLE_COLOR[1], IDLE_COLOR[2], 1);
        eids[i] = eid;
    }
    return eids;
}

/** the handle entities' world positions — the device gate reads this to assert the handle's `y` equals
 *  `heightAtCpu` at its `(x, z)` after an edit. */
export function handlePositions(): [[number, number, number], [number, number, number]] {
    if (handleEids[0] < 0 || handleEids[1] < 0) {
        throw new Error("roads: handles not yet created");
    }
    return [
        [
            Transform.pos.x.get(handleEids[0]),
            Transform.pos.y.get(handleEids[0]),
            Transform.pos.z.get(handleEids[0]),
        ],
        [
            Transform.pos.x.get(handleEids[1]),
            Transform.pos.y.get(handleEids[1]),
            Transform.pos.z.get(handleEids[1]),
        ],
    ];
}

const EditSystem: System = {
    name: "roads-edit",
    group: "simulation",
    annotations: { mode: "always" },

    update(state) {
        liveState = state;
        // find the canvas-presenting camera once
        if (camEid < 0) {
            for (const eid of state.query([Camera])) {
                if (state.has(eid, Transform)) {
                    camEid = eid;
                    break;
                }
            }
        }
        if (camEid < 0) return;

        // create the handle entities once the sphere mesh is registered
        if (handleEids[0] < 0 && Meshes.id("sphere") !== undefined) {
            handleEids = createHandles(state);
        }
        if (handleEids[0] < 0) return;

        // read the live document and place the handles at its endpoints (y = heightAtCpu)
        const doc = getDocument();
        const line = doc.polylines[0];
        const perm = makePermutation(SEED);
        for (let i = 0; i < 2; i++) {
            const [x, z] = line.points[i];
            const y = heightAtCpu(x, z, perm);
            Transform.pos.set(handleEids[i], x, y, z, 0);
        }

        // hover test: cursorRay → raySphere against both handle spheres
        const ray = cursorRay(state, camEid);
        hovered = -1;
        if (ray) {
            for (let i = 0; i < 2; i++) {
                const hx = Transform.pos.x.get(handleEids[i]);
                const hy = Transform.pos.y.get(handleEids[i]);
                const hz = Transform.pos.z.get(handleEids[i]);
                if (
                    raySphere(
                        ray.origin[0],
                        ray.origin[1],
                        ray.origin[2],
                        ray.dir[0],
                        ray.dir[1],
                        ray.dir[2],
                        hx,
                        hy,
                        hz,
                        HANDLE_RADIUS,
                    ) !== null
                ) {
                    hovered = i;
                    break;
                }
            }
        }

        // install the OrbitPick.claim once — the closure reads live Transform positions at call time, so
        // it stays correct across edits without re-installation. The claim reads the latch when a drag is
        // held (so orbit stays suppressed for the whole drag even when the cursor leaves the handle), and
        // runs a fresh hover test only when no drag is held.
        if (!claimInstalled) {
            OrbitPick.claim = () => {
                if (!liveState || camEid < 0 || handleEids[0] < 0) return false;
                if (grab.dragging) return true;
                const claimRay = cursorRay(liveState, camEid);
                if (!claimRay) return false;
                for (let i = 0; i < 2; i++) {
                    const hx = Transform.pos.x.get(handleEids[i]);
                    const hy = Transform.pos.y.get(handleEids[i]);
                    const hz = Transform.pos.z.get(handleEids[i]);
                    if (
                        raySphere(
                            claimRay.origin[0],
                            claimRay.origin[1],
                            claimRay.origin[2],
                            claimRay.dir[0],
                            claimRay.dir[1],
                            claimRay.dir[2],
                            hx,
                            hy,
                            hz,
                            HANDLE_RADIUS,
                        ) !== null
                    ) {
                        return true;
                    }
                }
                return false;
            };
            claimInstalled = true;
        }

        // grab latch: the orbit button is left (button 0). `stepGrab` latches on the rising edge over a
        // handle and holds until the falling edge — hover feeds only the colour, never the drag.
        grab = stepGrab(grab, Inputs.mouse.left, hovered);
        const { dragging, dragEnd } = grab;

        // drag: march the cursor ray against the flattened field, clamp to bounds, clamp to the
        // ROAD_MIN_LENGTH floor, and apply. When the march returns null (the ray misses the surface),
        // the drag holds its last valid target instead of skipping the frame.
        if (!dragging) {
            lastValidTarget = null;
        }
        if (dragging && ray) {
            const { segments, cutDepth } = buildNetworkGeometry(doc, SEED);
            const falloff = computeFalloff(cutDepth);
            const natural = (x: number, z: number) => heightAtCpu(x, z, perm);
            const hit = marchFlattenField(ray, segments, falloff, natural);
            if (hit) lastValidTarget = hit;
            if (lastValidTarget) {
                const [cx, cz] = clampToBound(lastValidTarget[0], lastValidTarget[1]);
                const [fx, fz] = clampDragTarget(doc, dragEnd, cx, cz);
                const newDoc = applyEdit(doc, dragEnd, fx, fz);
                void editDocument(newDoc).catch((err) => {
                    console.error("roads: edit failed", err);
                });
            }
        }

        // update handle colours based on hover / drag state
        for (let i = 0; i < 2; i++) {
            const color =
                dragging && i === dragEnd
                    ? GRABBED_COLOR
                    : i === hovered
                      ? HOVER_COLOR
                      : IDLE_COLOR;
            Color.rgba.set(handleEids[i], color[0], color[1], color[2], 1);
        }
    },

    dispose() {
        OrbitPick.claim = undefined;
        handleEids = [-1, -1];
        camEid = -1;
        liveState = null;
        hovered = -1;
        grab = createGrabState();
        claimInstalled = false;
        lastValidTarget = null;
    },
};

const RoadsEditPlugin: Plugin = {
    name: "RoadsEdit",
    systems: [EditSystem],
};

export default RoadsEditPlugin;
