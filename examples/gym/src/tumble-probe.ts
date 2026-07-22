// The interaction + visual standing probe for the tumble sample host (spec tumble-inline stage 5). Two
// param-gated phases the gym harness runs after the gold assert, on the live (still-stepping) scene — so
// they drive it rather than reading a settled snapshot, and NEVER feed the gold oracle:
//
//   • interact — a REAL pointer drag. It projects a grabbable dynamic body to a canvas pixel, dispatches
//     genuine `PointerEvent`s (pointerdown/move/up) at that pixel through the app's own InputPlugin
//     listeners, and lets the scenario's OWN fixed stepper turn them into a grab (cursorRay → beginGrab →
//     spring motor joint). It then reads the world back: the grabbed body moved (`grab-drag`), and — on a
//     settled stack — a light held-still grab did not launch the stack (`grab-no-launch`, the stage-3
//     grab-regression bound proven on a real device rather than headless).
//   • visual — a presence walk: the world's own `world.draw` produces nonzero joint gizmo output
//     (`visual-gizmos`), and the stage-4b native solid layer has materialized instanced `Part` entities
//     (`visual-solids`). Positive assertions on the derived visual layer, not absence-of-error.
//   • draws — the RENDER-level guard: reads the Part pack's own `drawArgs` back and asserts every distinct
//     solid mesh actually rasterizes on the GPU (`visual-draws`). `visual-solids` sees the Part ENTITY;
//     `visual-draws` sees the DRAW — so a GPU-side drop (a cull regression / degenerate bound / NaN in the
//     pack) that leaves the entity intact but draws nothing trips this while `visual-solids` stays green.
//
// The pointer path is dispatched-event, not CDP-trusted input: `shallot verify` drives the page through the
// published `window.__harness` and exposes no page.mouse to a caller, so a standing gate that stays a thin
// verify wrapper (bench:tumble's shape) synthesizes the events in-page. They are real `PointerEvent`
// objects on the real canvas element flowing through the real InputPlugin handlers — the same path a user's
// mouse takes, minus the browser's trusted-input flag.

import { Camera, Compute, Part, type State, Transform } from "@dylanebert/shallot";
import { Parts } from "@dylanebert/shallot/part/core";
import { screenToRay } from "@dylanebert/shallot/physics/core";
import {
    type Body,
    BodyType,
    type DebugDraw,
    defaultDebugDraw,
    type Mesh,
    type World,
    type WorldTransform,
} from "@dylanebert/shallot/tumble/core";
import { type Check, frames } from "./gym";
import type { SampleGold } from "./tumble-oracle";
import type { OverlayLayer } from "./tumble-overlay";
import { worldToScreen } from "./tumble-project";

/** what the tumble host hands the probe: the live world + camera + canvas, enough to project, aim, and
 *  drive the grab through the real input path. */
export interface ProbeContext {
    state: State;
    world: World;
    cam: number;
    canvas: HTMLCanvasElement;
    gold: SampleGold;
    /** the sample's overlay channel (null when the sample has no `render`) — the `overlay` phase reads its
     *  label/text counts as the demonstration-layer presence signal. */
    overlay: OverlayLayer | null;
    /** the live grab, when one is held (spec 6b/F1′): the floor-vanish instrument reads the kinematic anchor
     *  world position (is it below the ground plane?), the grabbed body pose, and the ray depth. Null when no
     *  grab is active. Instrumentation-only — never mutates the grab. */
    grab?: () => { anchor: number[]; target: number[]; body: number[]; depth: number } | null;
}

const FAR = 1000; // pick-ray length, matching the host's grab cast
// A launch would fling stack members metres in a few ticks; a weight-scaled hold keeps a settled stack put.
// Meaningful only alongside grab-drag in the same run: a grab that never engages also leaves the stack
// put, so no-launch alone is vacuous — never ship it as a scenario's sole interaction check.
const LAUNCH_LIMIT_M = 0.5;
// A grabbed body must travel at least this far past its own baseline motion under the drag. A settled
// scene's per-frame jitter is <0.01 m, so a 0.1 m floor is an order of magnitude above noise; a
// non-quiescent scene (a pendulum never sleeps) is covered by the measured baseline envelope the check
// adds on top. A free body clears metres; a hinge-constrained link travels less but still well past it.
const DRAG_MIN_M = 0.1;

/** run the param-gated probe phases and return their checks (`[]` when no probe opt is set, so the plain
 *  gold gate is unperturbed). `interact=drag` adds the settled-stack no-launch bound; any `interact` value
 *  runs the drag-moves-a-body check; `visual=1` adds the visual-presence checks; `draws=1` adds the
 *  render-level `visual-draws` guard (every solid mesh actually draws on the GPU); `overlay=1` adds the
 *  demonstration-layer presence check (nonzero projected `string3d` labels from the sample's `render()`). */
export async function runProbe(ctx: ProbeContext, opts: Record<string, unknown>): Promise<Check[]> {
    const checks: Check[] = [];
    if (opts.interact != null && opts.interact !== "") {
        checks.push(...(await drag(ctx, opts.interact === "drag")));
    }
    if (opts.visual != null && opts.visual !== "" && opts.visual !== "0") {
        checks.push(...visual(ctx));
    }
    if (opts.draws != null && opts.draws !== "" && opts.draws !== "0") {
        checks.push(...(await renderDraws(ctx)));
    }
    if (opts.overlay != null && opts.overlay !== "" && opts.overlay !== "0") {
        checks.push(...(await overlayPresence(ctx)));
    }
    return checks;
}

// --- the overlay-presence gate ---------------------------------------------------------------------------

/**
 * The demonstration-layer presence gate the engine fold's dropped `render()` overlays regressed. On a sample
 * with a projected-label overlay (events-joint-break hangs six threshold labels), the host's overlay channel
 * emits one `string3d` per label each frame; this reads the channel's OWN output back after a live frame and
 * asserts a nonzero label count — a positive presence assertion, not absence-of-error. A disconnected adapter
 * (no `render` wired) leaves `ctx.overlay` null and trips this red, the red-first shape.
 */
async function overlayPresence(ctx: ProbeContext): Promise<Check[]> {
    if (!ctx.overlay) {
        return [
            {
                name: "overlay-labels",
                pass: false,
                detail: "no overlay channel — the sample's render() is not wired",
            },
        ];
    }
    await frames(2); // let the draw system run the render hook against the current frame
    const emitted = ctx.overlay.labelCount();
    const shown = ctx.overlay.shownCount();
    const texts = ctx.overlay.textCount();
    return [
        {
            name: "overlay-labels",
            // post-cull count: emitted labels that project on-screen. Emit-count alone stays green when
            // the projection breaks (NaN/off-screen writes nothing visible) — assert what renders.
            pass: shown > 0,
            detail: `${shown} on-screen of ${emitted} emitted string3d label(s) + ${texts} HUD line(s)`,
            data: { shown, emitted, texts },
        },
    ];
}

// --- the rendered-draw presence gate ---------------------------------------------------------------------

/**
 * The render-LEVEL guard the CPU `visual-solids` check can't provide. `visual-solids` proves each world solid
 * has a live `Part` ENTITY, but a Part entity that exists can still fail to DRAW on the GPU — a frustum-cull
 * regression, a degenerate mesh bound, or a NaN reaching the pack drops the instance count to zero while the
 * entity survives untouched. So a floor that vanishes GPU-side leaves `visual-solids` green (the reported
 * defect class). This reads the Part pack's own `drawArgs` output back and asserts that at the deterministic
 * gold framing — which frames the whole scene, so every solid mesh is on-screen — **every distinct solid mesh
 * actually draws** (a pair with `instanceCount > 0`). If any mesh (the floor is its own pair) is silently
 * culled or dropped GPU-side, the drawing-pair count falls below the distinct-mesh count and this goes red —
 * including on hardware this repo's bridge can't reach, since the gate runs on the operator's own device.
 */
async function renderDraws(ctx: ProbeContext): Promise<Check[]> {
    await frames(2); // let the pack execute against the current (settled gold) frame before reading it back
    const meshes = distinctMeshCount(ctx.world);
    const drawing = await gpuDrawingPairs();
    if (drawing < 0) {
        return [
            {
                name: "visual-draws",
                pass: false,
                detail: "no GPU device / pack output to read back",
            },
        ];
    }
    return [
        {
            name: "visual-draws",
            pass: drawing === meshes,
            detail: `${drawing} GPU-drawing mesh pair(s) vs ${meshes} distinct solid mesh(es) (must be equal — every solid mesh draws at the gold framing)`,
            data: { drawing, meshes },
        },
    ];
}

// distinct solid geometries in the world, keyed exactly as the solid layer dedupes them (`tumble-solids.ts`
// `collectSolids`): one mesh per unique shape geometry. Read UNBOUNDED so a far body's mesh still counts.
function distinctMeshCount(world: World): number {
    const keys = new Set<object>();
    const H = 1e9;
    const dd: DebugDraw = {
        ...defaultDebugDraw(),
        drawingBounds: { lowerBound: { x: -H, y: -H, z: -H }, upperBound: { x: H, y: H, z: H } },
        drawShapes: true,
        drawSolidSphere: (_xf, sphere) => keys.add(sphere),
        drawSolidCapsule: (_xf, cap) => keys.add(cap),
        drawSolidHull: (_xf, hull) => keys.add(hull),
        drawSolidMesh: (_xf, mesh: Mesh) => keys.add(mesh.data),
    };
    world.draw(dd);
    return keys.size;
}

// count the Part pack's drawing pairs: a one-shot readback of `Parts.drawArgs` (COPY_SRC), tallying records
// whose `instanceCount` (lane 1 of the 20-byte DrawIndexedIndirect stride) is nonzero. The tumble host runs
// one shading camera and no shadow atlas, so only that view's slot carries nonzero counts — a whole-buffer
// scan is exactly its drawing-pair count. `-1` when no device / pack output exists yet.
async function gpuDrawingPairs(): Promise<number> {
    const device = Compute.device;
    const src = Parts.drawArgs;
    if (!device || !src) return -1;
    const staging = device.createBuffer({
        size: src.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, staging, 0, src.size);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const args = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    let drawing = 0;
    for (let i = 1; i < args.length; i += 5) if (args[i] > 0) drawing++;
    return drawing;
}

// --- the pointer drag ------------------------------------------------------------------------------------

async function drag(ctx: ProbeContext, checkLaunch: boolean): Promise<Check[]> {
    const { world, cam, canvas } = ctx;
    const rect = canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    if (w < 1 || h < 1) {
        return [{ name: "grab-drag", pass: false, detail: `canvas has no size (${w}×${h})` }];
    }
    const fovDeg = Camera.fov.get(cam);
    const near = Camera.near.get(cam);
    const camPos: [number, number, number] = [
        Transform.pos.x.get(cam),
        Transform.pos.y.get(cam),
        Transform.pos.z.get(cam),
    ];
    const camQuat: [number, number, number, number] = [
        Transform.rot.x.get(cam),
        Transform.rot.y.get(cam),
        Transform.rot.z.get(cam),
        Transform.rot.w.get(cam),
    ];

    // find a grabbable body: aim at each solid pose top-down; the first screen pixel whose forward-most hit
    // is a dynamic body is where a grab will land. Casting the pick ray directly (not via cursorRay) keeps
    // this input-independent; the stepper's cursorRay reconstructs the same ray from the same pixel.
    const aim = findGrabTarget(world, camPos, camQuat, fovDeg, near, w, h);
    if (!aim) {
        return [
            {
                name: "grab-drag",
                pass: false,
                detail: "no grabbable dynamic body on screen to grab",
            },
        ];
    }
    // baseline: watch the target over the drag's own observation window with no pointer down. A
    // non-quiescent scene moves the body on its own; the drag must beat that envelope, not ride it.
    const b0 = aim.body.getPosition();
    let baseline = 0;
    for (let i = 0; i < 20; i++) {
        await frames(1);
        const b = aim.body.getPosition();
        baseline = Math.max(baseline, Math.hypot(b.x - b0.x, b.y - b0.y, b.z - b0.z));
    }
    const p0 = aim.body.getPosition();

    // hover over the target, then press left. The stepper grabs on the button's down-edge (its next fixed
    // tick sees left && !prevLeft) using cursorRay at this same pixel — so it grabs `aim.body`.
    move(canvas, rect, aim.sx, aim.sy, 0);
    await frames(1);
    down(canvas, rect, aim.sx, aim.sy);
    await frames(4); // let the grab engage and hold

    const checks: Check[] = [];

    if (checkLaunch) {
        const before = poses(world);
        for (let i = 0; i < 8; i++) {
            move(canvas, rect, aim.sx, aim.sy, 1); // held still — same pixel, left down
            await frames(1);
        }
        const maxDisp = maxDisplacement(before, poses(world));
        checks.push({
            name: "grab-no-launch",
            pass: maxDisp < LAUNCH_LIMIT_M,
            detail: `held-still grab moved the stack ≤ ${maxDisp.toFixed(3)} m (< ${LAUNCH_LIMIT_M})`,
            data: { maxDisplacement: maxDisp },
        });
    }

    // drag the pointer up-and-across: the vertical lifts a free body clear, the horizontal swings a
    // hinge-constrained one (a pure vertical pull fights the hinge and barely moves a link). Read how far
    // the grabbed body travelled.
    const steps = 14;
    const dxTotal = 0.22 * w;
    const dyTotal = -0.22 * h; // screen −y = up
    for (let i = 1; i <= steps; i++) {
        move(canvas, rect, aim.sx + (dxTotal * i) / steps, aim.sy + (dyTotal * i) / steps, 1);
        await frames(1);
    }
    await frames(2);
    const p1 = aim.body.getPosition();
    const moved = Math.hypot(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
    up(canvas, rect, aim.sx + dxTotal, aim.sy + dyTotal);
    await frames(2);

    checks.push({
        name: "grab-drag",
        pass: moved > baseline + DRAG_MIN_M,
        detail: `grabbed body travelled ${moved.toFixed(3)} m under the pointer drag (baseline ${baseline.toFixed(3)} m + ${DRAG_MIN_M})`,
        data: { moved, baseline },
    });
    return checks;
}

interface Grabbable {
    sx: number;
    sy: number;
    body: Body;
}

// project every solid pose to a pixel (highest bodies first — a top box / link is unobstructed), and cast
// the pick ray back through it. The first pixel whose closest hit is a dynamic body is a valid grab aim.
function findGrabTarget(
    world: World,
    camPos: [number, number, number],
    camQuat: [number, number, number, number],
    fovDeg: number,
    near: number,
    w: number,
    h: number,
): Grabbable | null {
    const candidates = poses(world)
        .map((p) => ({ p, s: worldToScreen(camPos, camQuat, fovDeg, w, h, p) }))
        .filter((c) => c.s.front && c.s.x >= 0 && c.s.x <= w && c.s.y >= 0 && c.s.y <= h)
        .sort((a, b) => b.p.y - a.p.y);
    for (const c of candidates) {
        const ray = screenToRay(c.s.x, c.s.y, w, h, fovDeg, near, camPos, camQuat);
        const r = world.castRayClosest(
            { x: ray.origin[0], y: ray.origin[1], z: ray.origin[2] },
            { x: ray.dir[0] * FAR, y: ray.dir[1] * FAR, z: ray.dir[2] * FAR },
        );
        if (r.hit && r.shape && r.shape.getBody().getType() === BodyType.Dynamic) {
            return { sx: c.s.x, sy: c.s.y, body: r.shape.getBody() };
        }
    }
    return null;
}

// --- the visual-presence walk ----------------------------------------------------------------------------

function visual(ctx: ProbeContext): Check[] {
    const { state, world } = ctx;

    // gizmo layer: the world's own draw walk, joints only (drawShapes off), counting segment/point output.
    let segments = 0;
    const gizmo: DebugDraw = {
        ...defaultDebugDraw(),
        drawShapes: false,
        drawJoints: true,
        drawSegment: () => {
            segments++;
        },
        drawPoint: () => {
            segments++;
        },
        drawTransform: () => {
            segments++;
        },
    };
    world.draw(gizmo);

    // solid layer: the stage-4b native pool renders one Part per drawn shape each frame — count live Parts.
    let parts = 0;
    for (const _ of state.query([Part])) parts++;

    // the exact derived invariant: the derivation is TOTAL, so the live-Part count must EQUAL the world's own
    // solid-shape count — read here unbounded (see solidCount) so a body dragged far still counts. `parts > 0`
    // passed with a body silently missing; equality catches the manual-pass defect (a solid clipped out of the
    // derivation under interaction). Visuals never feed the gold oracle, so this can't perturb it.
    const solids = solidCount(world);

    return [
        {
            name: "visual-solids",
            pass: parts === solids,
            detail: `${parts} instanced Part(s) vs ${solids} world solid bodies (must be equal — total derivation)`,
            data: { parts, solids },
        },
        {
            name: "visual-gizmos",
            pass: segments > 0,
            detail: `${segments} joint-gizmo primitive(s) from the debug-draw walk`,
            data: { segments },
        },
    ];
}

// --- geometry + event helpers ----------------------------------------------------------------------------

/** the world's own solid-shape count, read UNBOUNDED (a ±1e9 draw box) so a body far from the origin still
 *  counts — the source of truth the derived Part layer must equal for the derivation to be total. */
function solidCount(world: World): number {
    let n = 0;
    const H = 1e9;
    const dd: DebugDraw = {
        ...defaultDebugDraw(),
        drawingBounds: { lowerBound: { x: -H, y: -H, z: -H }, upperBound: { x: H, y: H, z: H } },
        drawShapes: true,
        drawSolidSphere: () => {
            n++;
        },
        drawSolidCapsule: () => {
            n++;
        },
        drawSolidHull: () => {
            n++;
        },
        drawSolidMesh: () => {
            n++;
        },
    };
    world.draw(dd);
    return n;
}

/** every solid body pose the walk draws, in the walk's stable traversal order (creation order). Positions
 *  only — enough to project an aim and to diff a launch. */
function poses(world: World): { x: number; y: number; z: number }[] {
    const out: { x: number; y: number; z: number }[] = [];
    const push = (xf: WorldTransform): void => {
        out.push({ x: xf.p.x, y: xf.p.y, z: xf.p.z });
    };
    const dd: DebugDraw = {
        ...defaultDebugDraw(),
        drawShapes: true,
        drawSolidSphere: (xf) => push(xf),
        drawSolidCapsule: (xf) => push(xf),
        drawSolidHull: (xf) => push(xf),
        drawSolidMesh: (xf) => push(xf),
    };
    world.draw(dd);
    return out;
}

/** the largest per-body displacement between two pose snapshots (stable index correspondence — the walk
 *  order is fixed while no body is destroyed). */
function maxDisplacement(
    a: { x: number; y: number; z: number }[],
    b: { x: number; y: number; z: number }[],
): number {
    let max = 0;
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) {
        max = Math.max(max, Math.hypot(a[i].x - b[i].x, a[i].y - b[i].y, a[i].z - b[i].z));
    }
    return max;
}

function pointer(
    type: string,
    rect: DOMRect,
    sx: number,
    sy: number,
    buttons: number,
    button: number,
): PointerEvent {
    return new PointerEvent(type, {
        bubbles: true,
        cancelable: true,
        pointerId: 1,
        pointerType: "mouse",
        button,
        buttons,
        clientX: rect.left + sx,
        clientY: rect.top + sy,
    });
}

// dispatch on the canvas: the canvas listener (pointerHover → mouse.x/y) fires, and the event bubbles to
// the window listeners (pointerMove/Up → button state) — exactly a real mouse move/press over the canvas.
function move(
    canvas: HTMLCanvasElement,
    rect: DOMRect,
    sx: number,
    sy: number,
    buttons: number,
): void {
    canvas.dispatchEvent(pointer("pointermove", rect, sx, sy, buttons, -1));
}
function down(canvas: HTMLCanvasElement, rect: DOMRect, sx: number, sy: number): void {
    canvas.dispatchEvent(pointer("pointerdown", rect, sx, sy, 1, 0));
}
function up(canvas: HTMLCanvasElement, rect: DOMRect, sx: number, sy: number): void {
    canvas.dispatchEvent(pointer("pointerup", rect, sx, sy, 0, 0));
}
