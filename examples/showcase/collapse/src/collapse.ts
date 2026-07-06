import {
    Body,
    Color,
    Compute,
    mountOverlay,
    Part,
    type Plugin,
    ShapeKind,
    type State,
    type System,
} from "@dylanebert/shallot";
import { Physics } from "@dylanebert/shallot/physics/core";
import { Meshes } from "@dylanebert/shallot/render/core";

// Large collapsing AVBD structure — the spectacle proof of the rigid-body solver (roadmap "Showcases").
// A manifest project: scenes/collapse.scene authors the environment (lights, ground, orbit camera),
// shallot.json enables physics + this plugin, and the wall is spawned imperatively in `warm` below (a
// placement loop beats hundreds of <a> tags). The wall is a grid-cell weave —
// long along Z, tall, THIN through its X depth — of thick vertical POSTS tied by a horizontal FLOOR grillage,
// the floor planks overhanging the perimeter posts so the edge pieces sit flush. A ROW of compact balls rolls
// LOW from the +Z end along the wall's whole length at the base, knocking out the entire lower layer, so the
// storeys above pancake down with nothing left to stand on — a collapse that travels behind them. Posts are
// thick for a reliable settle. Runs on the default plugins + capacity (idiomatic) plus PhysicsPlugin /
// MirrorPlugin / ProfilePlugin; scalable by the knobs below.
//
// The launch writes the balls' velocity straight into the eid-indexed `bodies` SoA (`Physics.step.bodies`,
// the gym's seed path) — the solver authors a spawn pose + mass, not a live impulse. Reset rewrites every
// plank's pose/velocity columns the same way (mass/shape columns are seed-owned, left untouched), re-posing
// the wall without rebuilding it. The static ground lives in the scene at a lower eid, so the contiguous
// reset block is just the planks + balls.

const GROUND_TOP = 0.5; // ground is a static box half-Y 0.5, top face at y = 0.5

// ── scale: one knob ──────────────────────────────────────────────────────────
// TARGET block count. The wall sizes itself to hit it: the DEPTH (width) is fixed, the HEIGHT grows
// ~logarithmically (slow growth keeps it framed), and the LENGTH takes up the rest (it runs off-screen at
// high counts — the wall stretches into the distance, height stays in view). Keep TARGET under the default
// capacity (65536); the scene ground is sized to cover up to it.
const TARGET = 10000;

const PITCH = 1.9; // post grid spacing
const NX = 9; // posts through the DEPTH (width) — fixed across counts; a chunky wall, not a thin ribbon
const STOREYS = Math.max(4, Math.round(1.1 * Math.log(TARGET))); // height ~ log(count): reads tall, stays framed
// length fills the rest. count ≈ STOREYS·(NZ·(3·NX−1) − NX) (posts + both floor grillage layers) → invert for NZ:
const NZ = Math.max(2, Math.round((TARGET / STOREYS + NX) / (3 * NX - 1)));
const COUNT = STOREYS * (3 * NX * NZ - NZ - NX); // total planks — drives the ball mass

// post — a thick vertical column (full 0.6 x 1.4 x 0.6).
const POST: [number, number, number] = [0.3, 0.7, 0.3];
// floor grillage — short planks each bridging ONE cell (two adjacent posts), a layer across the depth then a
// layer along the length, the outermost planks overhanging the edge posts so the perimeter pieces sit flush.
// Short spans keep each plank under the per-body contact cap; the overlap couples the wall so a dropped slice
// drags its neighbours.
const FT = 0.13; // floor plank half-thickness (full 0.26)
const FW = 0.32; // floor plank half-width
const GAP = 0.005; // pre-space stacked layers a hair so the wall settles onto itself, not in penetration

const wallHalfZ = ((NZ - 1) / 2) * PITCH + POST[0]; // half the wall's length (Z), centred on the origin

// iron balls — a ROW across the wall's DEPTH (X), rolling the whole length at the base together so the
// collapse travels behind them, knocking out the lower layer so the storeys above pancake. Compact by
// constraint: the solver tracks at most PAIRS_PER_BODY = 8 contacts per body (broadphase keeps the nearest-K,
// drops the rest), so a body overlapping many more neighbours than that loses contacts and tunnels. A
// radius-~8 ball overlaps hundreds of blocks at once → far past the cap → tunnels; a compact ball overlaps a
// handful, so every contact is tracked. The row spans the depth a single big ball would; per-ball mass keeps
// the same total heft, and speed stays sub-pitch-per-frame so this is never speed-tunnelling either.
const BALL_R = 1.3; // compact — overlaps a handful of blocks, never near the per-body cap
const BALL_GAP = 0.4; // surface gap between resting balls so the row settles without mutual contact
const BALL_PITCH = 2 * BALL_R + BALL_GAP; // centre spacing across the depth
const wallDepth = (NX - 1) * PITCH + 2 * POST[0]; // the wall's full X extent (outer post faces)
const NBALLS = Math.max(2, Math.floor(wallDepth / BALL_PITCH) + 1); // enough to span the depth
const BALL_MASS = Math.max(40, Math.round((COUNT * 0.3) / NBALLS)); // total heft ≈ the old single ball
const BALL_SPEED = Math.min(100, Math.max(50, (wallHalfZ + BALL_R + 2) / 1.4));
const LAUNCH_VEL: [number, number, number] = [0, 0, -BALL_SPEED]; // −z along the length, low

// solver body columns (SoA `bodies[col*eidCap + eid]`): pose + velocity state. Cols 9/10/11 (moment·mass,
// halfExtents·friction, shape·radius) are seed-owned and never rewritten here.
const POS_COLS = [0, 2, 4]; // posLin, inertialLin, initialLin — all the step-start pose
const QUAT_COLS = [1, 3, 5]; // posAng, inertialAng, initialAng
const VEL_COLS = [6, 7, 8]; // velLin, velAng, prevVelLin — zeroed

const clamp01 = (x: number): number => Math.min(1, Math.max(0, x));
const frac = (x: number): number => x - Math.floor(x);
const rnd = (i: number): number => frac(Math.sin(i * 127.1 + 311.7) * 43758.5453);

// OKLCH (perceptual lightness 0..1, chroma, hue degrees) → LINEAR sRGB — the space Color.rgba stores
// (the FS shades `lit(color, n)` in linear, glaze encodes sRGB at present). Gamut-clamped to [0,1].
// Authoring colour perceptually keeps the gradient's lightness + chroma even as the hue sweeps.
function oklch(L: number, C: number, hDeg: number): [number, number, number] {
    const h = (hDeg * Math.PI) / 180;
    const a = C * Math.cos(h);
    const b = C * Math.sin(h);
    const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
    const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
    const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
    return [
        clamp01(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
        clamp01(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
        clamp01(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s),
    ];
}

interface Scene {
    firstEid: number; // the planks + balls occupy the contiguous range [firstEid, firstEid + count)
    count: number;
    ballFirst: number; // first ball eid; the NBALLS balls occupy [ballFirst, ballFirst + NBALLS)
    initPos: Float32Array<ArrayBuffer>; // authored pose in eid order, for reset (count * 4)
    initQuat: Float32Array<ArrayBuffer>;
    zeros: Float32Array<ArrayBuffer>;
    ready: boolean;
}

const scene: Scene = {
    firstEid: -1,
    count: 0,
    ballFirst: -1,
    initPos: new Float32Array(0),
    initQuat: new Float32Array(0),
    zeros: new Float32Array(0),
    ready: false,
};

// ── scene assembly ──────────────────────────────────────────────────────────

// a Body+Part box. `half` is its half-extents; `mass <= 0` is static. A post is a tall half, a floor plank a
// flat one — orientation lives in halfExtents, so every body spawns axis-aligned (no quaternion needed).
function box(
    state: State,
    pos: [number, number, number],
    half: [number, number, number],
    mass: number,
    color: [number, number, number],
): number {
    const eid = state.create();
    state.add(eid, Body);
    Body.pos.set(eid, pos[0], pos[1], pos[2], 0);
    Body.halfExtents.set(eid, half[0], half[1], half[2], 0);
    Body.mass.set(eid, mass);
    state.add(eid, Part);
    state.add(eid, Color);
    Color.rgba.set(eid, color[0], color[1], color[2], 1);
    return eid;
}

// spawn the wall + balls into the live State — runs from the plugin's `warm` (post-scene), idempotent: a
// State rebuild re-runs it, re-creating the derived bodies (they live in State, not the Document).
function build(state: State): void {
    // no per-scene solver config — the engine's ship default (iters=6) settles the 10-storey wall, so this
    // runs on the plain PhysicsPlugin defaults (physics.md "iters is a free knob"). A taller TARGET that
    // pancakes is the one case to bump iters per-scene via `Physics.step.configure`.

    const [pt, ph] = POST; // post half-thickness, post half-height
    const Sy = 2 * ph + 4 * FT + 3 * GAP; // storey pitch: post + depth-floor + length-floor + 3 seating gaps
    const gx = (i: number): number => (i - (NX - 1) / 2) * PITCH; // depth coord (X), centred on the origin
    const gz = (j: number): number => (j - (NZ - 1) / 2) * PITCH; // length coord (Z), centred on the origin
    const Oh = pt; // floor planks overhang the outermost posts by a post-half, so edge pieces sit flush
    const maxY = STOREYS * Sy;

    let firstEid = -1; // the first plank — the start of the contiguous reset block (the scene's bodies precede it)
    // every block its own colour (free in shallot — per-entity Color): a cohesive sunset→sky gradient by
    // HEIGHT — warm amber at the base → cool blue at the top, held at an intentional chroma. Keyed to height
    // (not length) so it stays fully in frame as the length runs off-screen at high counts; a slow length-wise
    // lightness undulation + a per-block jitter keep neighbours distinct. Dark low-chroma ground lets it read.
    const blockColor = (z: number, y: number): [number, number, number] => {
        const t = clamp01(y / maxY);
        const drift = Math.sin(z * 0.06) * 0.04;
        const jitter = (rnd(z * 31.7 + y * 13.1) - 0.5) * 0.03;
        return oklch(0.6 + drift + jitter, 0.135, 55 - t * 155);
    };
    // a run of floor planks bridging adjacent posts along one axis, the two end planks overhanging the
    // perimeter posts (the issue-1 fix — a plank ending AT a post centre leaves the edge post half-supported).
    const floorRun = (along: "x" | "z", cross: number, y: number): void => {
        const n = along === "x" ? NX : NZ;
        const at = along === "x" ? gx : gz;
        for (let c = 0; c < n - 1; c++) {
            const a = at(c) - (c === 0 ? Oh : 0);
            const b = at(c + 1) + (c === n - 2 ? Oh : 0);
            const mid = (a + b) / 2;
            const hl = (b - a) / 2;
            const half: [number, number, number] = along === "x" ? [hl, FT, FW] : [FW, FT, hl];
            const pos: [number, number, number] = along === "x" ? [mid, y, cross] : [cross, y, mid];
            box(state, pos, half, 1, blockColor(along === "x" ? cross : mid, y));
        }
    };

    // the wall — built imperatively and consecutively so the planks form one contiguous eid block (the scene's
    // ground / camera / lights already exist at lower eids; the reset writes only this block). GROUND_TOP is
    // the scene ground's top face, which the storeys rest on.
    for (let k = 0; k < STOREYS; k++) {
        const baseY = GROUND_TOP + k * Sy;
        // posts — the storey's vertical columns, resting on the ground (k=0) or the storey below's floor
        for (let j = 0; j < NZ; j++)
            for (let i = 0; i < NX; i++) {
                const e = box(
                    state,
                    [gx(i), baseY + ph, gz(j)],
                    POST,
                    1,
                    blockColor(gz(j), baseY + ph),
                );
                if (firstEid < 0) firstEid = e;
            }
        // depth-floor — planks bridging the thin X depth, one run per length row, resting on the post tops
        const yX = baseY + 2 * ph + GAP + FT;
        for (let j = 0; j < NZ; j++) floorRun("x", gz(j), yX);
        // length-floor — planks along Z (the length) laid ON the depth-floor: ties the length + seats the next storey
        const yZ = baseY + 2 * ph + 2 * FT + 2 * GAP + FT;
        for (let i = 0; i < NX; i++) floorRun("z", gx(i), yZ);
    }
    scene.firstEid = firstEid;

    // the wrecking balls — sphere colliders (rounded narrowphase) resting ON the ground at the +Z end of the
    // wall (the camera-facing front), spread across the depth (X), launched on smash to roll the whole LENGTH
    // (−z) low along the base, knocking out the lower layer so the storeys above pancake down.
    const launchY = GROUND_TOP + BALL_R; // rest each ball on the ground — its centre a radius above the top face
    const launchZ = wallHalfZ + BALL_R + 2; // the +Z end, clear of the wall
    const [br, bg, bb] = oklch(0.8, 0.015, 250); // light cool chrome — reads against the blocks + ground
    let ballFirst = -1;
    for (let n = 0; n < NBALLS; n++) {
        const ball = state.create();
        if (ballFirst < 0) ballFirst = ball;
        state.add(ball, Body);
        Body.shape.set(ball, ShapeKind.Sphere);
        Body.pos.set(ball, (n - (NBALLS - 1) / 2) * BALL_PITCH, launchY, launchZ, 0);
        Body.halfExtents.set(ball, 0, 0, 0, BALL_R); // sphere: core point, radius rides .w
        Body.mass.set(ball, BALL_MASS);
        state.add(ball, Part);
        Part.mesh.set(ball, Meshes.id("sphere") ?? 0);
        state.add(ball, Color);
        Color.rgba.set(ball, br, bg, bb, 1);
    }
    scene.ballFirst = ballFirst;

    scene.count = ballFirst + NBALLS - firstEid; // planks + the ball row, one contiguous block
    captureInitial();

    armed = true;
    settleStart = -1;
    scene.ready = true;
}

// the authored pose in eid order — reset rewrites the live bodies back to this without a rebuild
function captureInitial(): void {
    const n = scene.count;
    scene.initPos = new Float32Array(n * 4);
    scene.initQuat = new Float32Array(n * 4);
    scene.zeros = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
        const eid = scene.firstEid + i;
        const o = i * 4;
        scene.initPos[o] = Body.pos.x.get(eid);
        scene.initPos[o + 1] = Body.pos.y.get(eid);
        scene.initPos[o + 2] = Body.pos.z.get(eid);
        scene.initQuat[o + 3] = 1; // identity quat (planks + ball spawn axis-aligned)
    }
}

// ── perturbation + reset (write the eid-indexed bodies SoA directly) ──────────

function writeCol(col: number, base: number, data: Float32Array<ArrayBuffer>): void {
    const step = Physics.step;
    if (!step || !Compute.device) return;
    Compute.device.queue.writeBuffer(step.bodies, (col * step.eidCap + base) * 16, data);
}

// teleport each ball back to its launch point and give it its launch velocity. Teleport (not just velocity)
// so a re-smash always fires from the start, wherever the balls came to rest after the last one.
export function smash(): void {
    if (!scene.ready) return;
    const q = new Float32Array([0, 0, 0, 1]);
    const z = new Float32Array(4);
    const vel = new Float32Array([LAUNCH_VEL[0], LAUNCH_VEL[1], LAUNCH_VEL[2], 0]);
    for (let n = 0; n < NBALLS; n++) {
        const eid = scene.ballFirst + n;
        const o = (eid - scene.firstEid) * 4;
        const p = new Float32Array([
            scene.initPos[o],
            scene.initPos[o + 1],
            scene.initPos[o + 2],
            0,
        ]);
        for (const c of POS_COLS) writeCol(c, eid, p);
        for (const c of QUAT_COLS) writeCol(c, eid, q);
        writeCol(6, eid, vel);
        writeCol(7, eid, z);
        writeCol(8, eid, z);
    }
    armed = false;
}

// reset doubles as replay: re-pose every body to its authored start at zero velocity (one writeBuffer per
// column over the block) and re-arm the one-shot launch, so the wall settles and the ball fires again. Mass /
// shape columns are untouched, so the seeded inertia survives — this is a re-pose, not a re-seed.
export function reset(): void {
    if (!scene.ready) return;
    for (const c of POS_COLS) writeCol(c, scene.firstEid, scene.initPos);
    for (const c of QUAT_COLS) writeCol(c, scene.firstEid, scene.initQuat);
    for (const c of VEL_COLS) writeCol(c, scene.firstEid, scene.zeros);
    armed = true;
    settleStart = -1;
}

// ── one-shot director: fire the ball once the wall has settled ────────────────

let armed = false;
let settleStart = -1; // fixedTick the settle countdown began (-1 = not started)
const SETTLE_TICKS = 210; // ~3.5 s for the wall to come fully to rest before the launch

const DirectorSystem: System = {
    name: "collapse-director",
    group: "fixed",
    update(state) {
        if (!scene.ready || !armed) return;
        const tick = state.time.fixedTick;
        if (settleStart < 0) settleStart = tick;
        if (tick - settleStart >= SETTLE_TICKS) smash();
    },
};

// the plugin: spawns the wall in `warm`, runs the one-shot director, owns the control panel. `warm` is
// post-scene + idempotent (ecs.md "Reload-safety"); `dispose` tears the panel down so a State rebuild
// (play/stop, plugin toggle) doesn't stack overlays.
let panelCleanup: (() => void) | null = null;

const CollapsePlugin: Plugin = {
    name: "Collapse",
    systems: [DirectorSystem],
    warm(state) {
        build(state);
        panelCleanup?.();
        panelCleanup = mountPanel();
    },
    dispose() {
        panelCleanup?.();
        panelCleanup = null;
    },
};

// the manifest references this module by path (`"Collapse": "./src/collapse"`) and imports its default
export default CollapsePlugin;

// ── control panel ─────────────────────────────────────────────────────────────

// mounted into the engine's sandboxed overlay (`mountOverlay`) — the canvas-bounded surface `config.ui`
// hands an app, the contract-correct home for plugin-owned DOM (it can't spill into the editor viewport).
function mountPanel(): () => void {
    const overlay = mountOverlay(document.querySelector("canvas"));
    const panel = document.createElement("div");
    panel.style.cssText = [
        "position:absolute",
        "top:16px",
        "right:16px",
        "display:flex",
        "flex-direction:column",
        "gap:10px",
        "padding:14px 16px",
        "background:rgba(12,10,9,0.62)",
        "backdrop-filter:blur(8px)",
        "border:1px solid rgba(255,255,255,0.08)",
        "border-radius:10px",
        "font:500 12px/1.4 'JetBrains Mono',monospace",
        "color:#e7e1da",
        "pointer-events:auto",
        "user-select:none",
    ].join(";");

    const title = document.createElement("div");
    title.textContent = "collapsing avbd";
    title.style.cssText =
        "font-size:11px;letter-spacing:0.08em;color:#9a948c;text-transform:uppercase";

    const row = document.createElement("div");
    row.style.cssText = "display:flex;gap:8px";

    const button = (label: string, onClick: () => void): void => {
        const b = document.createElement("button");
        b.textContent = label;
        b.style.cssText = [
            "flex:1",
            "padding:8px 16px",
            "background:rgba(255,255,255,0.06)",
            "border:1px solid rgba(255,255,255,0.12)",
            "border-radius:6px",
            "color:#e7e1da",
            "font:500 12px 'JetBrains Mono',monospace",
            "cursor:pointer",
        ].join(";");
        b.onmouseenter = () => {
            b.style.background = "rgba(255,255,255,0.12)";
        };
        b.onmouseleave = () => {
            b.style.background = "rgba(255,255,255,0.06)";
        };
        b.onclick = onClick;
        row.appendChild(b);
    };
    button("smash", smash);
    button("reset", reset);

    const hint = document.createElement("div");
    hint.textContent = "drag to orbit · scroll to zoom";
    hint.style.cssText = "font-size:10px;color:#79736b";

    panel.append(title, row, hint);
    overlay.appendChild(panel);
    return () => overlay.remove();
}
