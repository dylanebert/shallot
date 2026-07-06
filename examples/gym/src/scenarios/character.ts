import {
    AmbientLight,
    Body,
    Camera,
    CameraMode,
    Character,
    CharacterPlugin,
    Color,
    Compute,
    DirectionalLight,
    GlazePlugin,
    InputPlugin,
    type Mirror,
    MirrorPlugin,
    mirror,
    Orbit,
    OrbitPlugin,
    Part,
    PartPlugin,
    PhysicsPlugin,
    Player,
    PlayerPlugin,
    type Plugin,
    RenderPlugin,
    run,
    Sear,
    SearPlugin,
    ShapeKind,
    SlabPlugin,
    type State,
    type System,
    Transform,
    TransformsPlugin,
} from "@dylanebert/shallot";
import { grounded, jump, move, pose } from "@dylanebert/shallot/character/core";
import { Profile, ProfilePlugin } from "@dylanebert/shallot/extras";
import { BODY_VEC4, PENALTY_MIN, Physics } from "@dylanebert/shallot/physics/core";
import type { Vec3 } from "../../../../packages/shallot/tests/avbd/math";
import { type Check, frames, type Params, register, type Scenario, settle } from "../gym";
import { disposeLoad, getLoad, LoadPlugin, setLoad } from "../load";

// character — the kinematic capsule controller (Phase 6.4). A character is a `mass <= 0` capsule whose pose a
// CONTROLLER, not the solver, owns: the CPU sweep (`standard/character`, `CharacterSweepSystem`) collide-and-
// slides it against the scene each fixed step (along the geometric closest-point MTV) BEFORE the GPU solve,
// uploads the swept pose as a kinematic body (`setKinematic`), and the rigid solve runs with the character in
// the body list. The scene authors characters as `Character` COMPONENTS (`CharacterPlugin` registers every
// `[Character, Body]`) and drives them by eid through the `character/core` API (`move` / `jump` / `grounded`).
// The sweep's own correctness is the deterministic CPU oracle gate (`tests/avbd/character-sweep.oracle.ts`,
// the f64-parity tier); this real-GPU scene gates the integrated behavior + the input→camera latency the CPU
// sweep buys (there is no GPU character pass to single-step compare — it was deleted with the camera rewire).
//
// The gate set, by what each covers:
//   • the live behavioral scene (the integrated flow, rendered): drop-to-rest, wall-stop + box-undisturbed,
//     full-speed push (no overtake), push-into-wall (bounded, the legacy explosive case), slope hold-vs-slide.
//   • the latency probe (`probe=1`): position input→camera latency holds at the orientation (CPU same-frame)
//     floor — no GPU readback in the position path — under an induced GPU-load knob, decoupled from frame_time.
//   • measured — the per-step solver spans as structured `Check.data`, the bench seam.
//
// Like the pile scenario, the live view is the orbit camera + the HUD; interactive grab stays in the labs.

const G = -10;
const DT = 1 / 60;
const ALPHA = 0.99;
const BETA_LIN = 1e4;
const GAMMA = 0.999;
const MAX_COLORS = 8;
const ITERS = 10; // the gym's robustness config (decoupled from the shipped iters=6)
const HALF_H = 0.5; // capsule core half-height
const RADIUS = 0.3; // capsule radius → resting offset = HALF_H + RADIUS = 0.8 above a surface
const REST_OFFSET = HALF_H + RADIUS;
const MAX_SLOPE = 45; // the walkable cutoff: 30° holds, 60° slides
const JUMP_SPEED = 5; // → apex = JUMP_SPEED²/(2|G|) = 1.25 m above the launch

const cfg = {
    dt: DT,
    gravity: G,
    alpha: ALPHA,
    penalty: PENALTY_MIN,
    betaLin: BETA_LIN,
    gamma: GAMMA,
    iterations: ITERS,
    maxColors: MAX_COLORS,
};

// per-eid horizontal input (x/z) + the set of eids that spam jump. A fixed-group DriverSystem pushes these
// each tick through the character/core drive (how a game drives the controller); a char in neither idles.
const moves = new Map<number, [number, number]>();
const jumps = new Set<number>();

// captured eids by station (the character buffer is registered by CharacterPlugin; we drive + read by eid)
const ch: Record<string, number> = {};
let wallBoxEid = -1;
let pushBoxEid = -1;
let pinBoxEid = -1;
let pushTraj = "";
let slope30SpawnX = Number.NaN;
let slope60SpawnX = Number.NaN;
let bodyMirror: Mirror | null = null;

const DriverPlugin: Plugin = {
    name: "CharDriver",
    systems: [
        {
            name: "char-driver",
            group: "fixed",
            update() {
                if (!Physics.step) return;
                for (const [eid, [vx, vz]] of moves) move(eid, vx, vz);
                for (const eid of jumps) jump(eid); // spam: the single-jump gating lives in the controller
            },
        } satisfies System,
    ],
};

// ── latency probe ────────────────────────────────────────────────────────────
// The probe path (probe=1) is the permanent regression gate that position input→camera latency carries no
// GPU readback: the CPU sweep writes the character's pose in the fixed phase, BEFORE any GPU work, so the
// pose is same-frame like orientation (which `setLook`s in the sim group). It runs on a minimal floor+Player
// scene and counts ENGINE frames (not GPU timestamps — that sidesteps the timestamp-vs-fixed-step aliasing)
// for three quantities, all under an induced GPU-load knob calibrated into the felt-lag regime:
//   • ORIENTATION — set Player.yaw, frames until the camera Transform.rot reflects it (CPU same-frame floor);
//   • POSE (the gate) — hold a move, frames until `pose(eid)` (the CPU CharState the sweep owns) reflects it.
//     Same-frame (≈ orientation) ⇒ no fence/Mirror in the position path; a readback would push it to 2–3;
//   • CAMERA (reporter) — frames until Transform.pos reflects it (pose + the kept one-tick interpolation).
// The `load` compute pass burns GPU time so the fence wait grows and effective fps drops into the felt-lag
// regime (≥1 fixed step per render frame), where the fixed-step quantization that dominates input latency at
// high fps collapses and the pose-vs-orientation comparison is clean. `calibrateLoad` ramps it per-hardware.

let playerEid = -1;
let playerCamEid = -1;
let probeResult: Record<string, number> | null = null;
let probeParams: Params | null = null;
// the GPU-load knob is the shared `LoadPlugin` (../load): `setLoad(iters)` ramps the per-lane burn,
// `getLoad()` reads it. calibrateLoad drives it into the felt-lag regime; the `load` profiler span the
// plugin times is ignored here (the probe counts engine frames, not GPU timestamps).
const _probePose: [number, number, number] = [0, 0, 0]; // reused scratch for the injectPose CharState read

function staticBox(
    state: State,
    pos: Vec3,
    half: Vec3,
    color: [number, number, number],
    quat: [number, number, number, number] = [0, 0, 0, 1],
): number {
    const eid = state.create();
    state.add(eid, Body);
    Body.shape.set(eid, ShapeKind.Box);
    Body.pos.set(eid, pos[0], pos[1], pos[2], 0);
    Body.quat.set(eid, quat[0], quat[1], quat[2], quat[3]);
    Body.halfExtents.set(eid, half[0], half[1], half[2], 0);
    Body.mass.set(eid, 0);
    state.add(eid, Part);
    state.add(eid, Color);
    Color.rgba.set(eid, color[0], color[1], color[2], 1);
    return eid;
}

function dynBox(
    state: State,
    pos: Vec3,
    half: Vec3,
    mass: number,
    color: [number, number, number],
    friction = 0.4,
): number {
    const eid = state.create();
    state.add(eid, Body);
    Body.shape.set(eid, ShapeKind.Box);
    Body.pos.set(eid, pos[0], pos[1], pos[2], 0);
    Body.halfExtents.set(eid, half[0], half[1], half[2], 0);
    Body.mass.set(eid, mass);
    Body.friction.set(eid, friction);
    state.add(eid, Part);
    state.add(eid, Color);
    Color.rgba.set(eid, color[0], color[1], color[2], 1);
    return eid;
}

// a kinematic capsule authored as a Character component (CharacterPlugin registers it with the solver). The
// tuning (slope / jump) lives on Character; the capsule geometry on Body.
function capsule(state: State, pos: Vec3, color: [number, number, number], jumpSpeed = 0): number {
    const eid = state.create();
    state.add(eid, Body);
    Body.shape.set(eid, ShapeKind.Capsule);
    Body.pos.set(eid, pos[0], pos[1], pos[2], 0);
    Body.halfExtents.set(eid, 0, HALF_H, 0, RADIUS);
    Body.mass.set(eid, 0);
    Body.friction.set(eid, 0.8);
    state.add(eid, Character);
    Character.maxSlope.set(eid, MAX_SLOPE);
    Character.jumpSpeed.set(eid, jumpSpeed);
    state.add(eid, Part);
    state.add(eid, Color);
    Color.rgba.set(eid, color[0], color[1], color[2], 1);
    return eid;
}

const qzRot = (rad: number): [number, number, number, number] => [
    0,
    0,
    Math.sin(rad / 2),
    Math.cos(rad / 2),
];

const scenario: Scenario = {
    name: "character",
    params: [
        // the bench's sweep surface (scripts/physics-bench.ts): N inert far statics. The CPU sweep's cull
        // rejects every one (out of reach of all stations), so every behavioral gate stays valid at any
        // filler — the world-body-count load on the sweep's per-character gather scan, swept for robustness.
        {
            key: "filler",
            type: "number",
            default: 0,
            min: 0,
            max: 16384,
            step: 256,
            rebuild: true,
            when: (v) => v.probe !== true,
        },
        // the input→camera latency probe (roadmap "CPU character controller"). probe=1 swaps the 6-station
        // scene for a minimal one (floor + a Player capsule + a first-person camera) and gates that the
        // position input→camera latency holds at the orientation (CPU same-frame) floor — no GPU readback —
        // even under load; `load` is the GPU-time inflation knob (compute workgroups burned per frame) that
        // drops effective fps into the felt-lag regime, self-calibrated by the probe.
        { key: "probe", type: "bool", default: false, rebuild: true },
        {
            key: "load",
            type: "number",
            default: 0,
            min: 0,
            max: 65535,
            step: 1024,
            when: (v) => v.probe === true,
        },
    ],

    async build(_canvas, p: Params) {
        probeResult = null;
        if (p.probe) return buildProbe(p);
        const filler = (p.filler as number) | 0;
        const { state, dispose } = await run({
            defaults: false,
            capacity: 64 + filler,
            plugins: [
                ProfilePlugin,
                SlabPlugin,
                MirrorPlugin,
                TransformsPlugin,
                InputPlugin,
                OrbitPlugin,
                RenderPlugin,
                PhysicsPlugin,
                CharacterPlugin,
                DriverPlugin,
                PartPlugin,
                SearPlugin,
                GlazePlugin,
            ],
        });

        state.add(state.create(), AmbientLight);
        state.add(state.create(), DirectionalLight);
        Physics.step?.configure(cfg);

        // one shared flat floor for the drop / wall / push stations, laid out in z-lanes. top at y = 0.5, so
        // a settled capsule rests at y = 1.3.
        staticBox(state, [0, 0, 0], [40, 0.5, 40], [0.28, 0.3, 0.34]);
        const restY = 0.5 + REST_OFFSET;

        // drop-to-rest — a capsule released above the floor falls, settles, grounded. No input.
        ch.drop = capsule(state, [-12, 3, -12], [0.45, 0.75, 0.55]);

        // wall-stop + box undisturbed — a capsule walks +x into a tall static wall (near face x = -2); the
        // sweep stops it, a dynamic box BEHIND it stays undisturbed (kin-static contacts never escalate).
        staticBox(state, [0, 4, -8], [2, 4, 2], [0.3, 0.32, 0.36]);
        ch.wall = capsule(state, [-6, restY, -8], [0.8, 0.5, 0.45]);
        wallBoxEid = dynBox(state, [-9, restY - 0.3, -8], [0.4, 0.4, 0.4], 1, [0.85, 0.55, 0.35]);

        // push a dynamic box — a capsule walks +x at FULL speed into a tall low-friction crate and shoves it
        // forward (velocity-transfer push), staying behind it (no overtake/tunnel).
        ch.push = capsule(state, [-1.7, restY, -4], [0.5, 0.6, 0.85]);
        pushBoxEid = dynBox(state, [-1, restY, -4], [0.4, 0.8, 0.4], 0.3, [0.85, 0.7, 0.4], 0.05);

        // push a box INTO a wall — the legacy explosive case; asserted only finite + bounded (it may squirt out).
        staticBox(state, [4, 1, 0], [0.5, 4, 2], [0.32, 0.3, 0.3]);
        ch.pushWall = capsule(state, [0, restY, 0], [0.85, 0.45, 0.45]);
        pinBoxEid = dynBox(
            state,
            [2.4, restY - 0.3, 0],
            [0.4, 0.4, 0.4],
            0.6,
            [0.8, 0.4, 0.5],
            0.05,
        );

        // jump — a capsule that spams the jump button (DriverSystem calls jump() each tick), so it bounces.
        ch.jumper = capsule(state, [-12, restY, 4], [0.55, 0.55, 0.9], JUMP_SPEED);

        // slope hold vs slide — a 30° ramp (walkable: holds) + a 60° ramp (too steep: slides), each a rotated
        // static box with a capsule dropped onto its top face.
        const slopeChar = (
            deg: number,
            cx: number,
            color: [number, number, number],
        ): { eid: number; spawnX: number } => {
            const a = (deg * Math.PI) / 180;
            const n: Vec3 = [-Math.sin(a), Math.cos(a), 0];
            staticBox(state, [cx, 5, 12], [6, 0.5, 6], [0.3, 0.33, 0.3], qzRot(a));
            const top: Vec3 = [
                cx + n[0] * (0.5 + REST_OFFSET),
                5 + n[1] * (0.5 + REST_OFFSET) + 1.2,
                12,
            ];
            return { eid: capsule(state, top, color), spawnX: top[0] };
        };
        const s30 = slopeChar(30, 14, [0.5, 0.8, 0.55]);
        const s60 = slopeChar(60, 24, [0.85, 0.5, 0.5]);
        ch.slope30 = s30.eid;
        ch.slope60 = s60.eid;
        slope30SpawnX = s30.spawnX;
        slope60SpawnX = s60.spawnX;

        // filler: inert far statics (Body only, no Part) in a grid past z = 120, out of every station's
        // reach — pure world-body-count load on the CPU sweep's per-character gather scan
        const side = Math.ceil(Math.sqrt(Math.max(filler, 1)));
        for (let i = 0; i < filler; i++) {
            const eid = state.create();
            state.add(eid, Body);
            Body.shape.set(eid, ShapeKind.Box);
            Body.pos.set(eid, ((i % side) - side / 2) * 2, 0.5, 120 + Math.floor(i / side) * 2, 0);
            Body.halfExtents.set(eid, 0.5, 0.5, 0.5, 0);
            Body.mass.set(eid, 0);
        }

        moves.set(ch.wall, [2, 0]); // walk into the wall
        moves.set(ch.push, [3, 0]); // full walking speed — the no-overtake assert gates the push
        moves.set(ch.pushWall, [2, 0]); // shove the pin box into the wall
        jumps.add(ch.jumper); // spam jump — single jump per landing (the controller gates it)

        const cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Camera);
        state.add(cam, Sear);
        state.add(cam, Orbit);
        Camera.mode.set(cam, CameraMode.Perspective);
        Orbit.yaw.set(cam, Math.PI / 6);
        Orbit.pitch.set(cam, Math.PI / 10);
        Orbit.distance.set(cam, 38);

        await frames(3);
        if (Physics.step) bodyMirror = mirror(Physics.step.bodies);

        // run + sample across the window — ~3 s of sim (headless rAF rate × fixed step 60 Hz). Record the push
        // char/box trajectory (the carry, surfaced in the push detail). The slope check reads only the settled
        // final pose vs the build-time spawn x, so it's independent of how much sim-time elapses per frames(30)
        // (the headless rAF rate differs across platforms — a mid-window landed-x sample is not portable).
        const traj: string[] = [];
        for (let k = 0; k < 12; k++) {
            await frames(30);
            if (!bodyMirror?.snapshot) continue;
            const s = new Float32Array(bodyMirror.snapshot.bytes);
            const cap = s.length / (BODY_VEC4 * 4);
            traj.push(
                `char ${s[(0 * cap + ch.push) * 4].toFixed(1)}/box ${s[(0 * cap + pushBoxEid) * 4].toFixed(1)}`,
            );
        }
        pushTraj = traj.join(" ");

        return {
            state,
            dispose() {
                bodyMirror?.dispose();
                bodyMirror = null;
                moves.clear();
                jumps.clear();
                dispose();
            },
        };
    },

    async assert(): Promise<Check[]> {
        if (probeResult) return probeChecks();
        const checks: Check[] = [];
        checks.push(...(await liveGates()));
        checks.push(await measured());
        return checks;
    },

    live(): string {
        if (!bodyMirror?.snapshot) return "character — warming";
        const s = new Float32Array(bodyMirror.snapshot.bytes);
        const cap = s.length / (BODY_VEC4 * 4);
        const p = (eid: number): string =>
            `${s[(0 * cap + eid) * 4].toFixed(1)},${s[(0 * cap + eid) * 4 + 1].toFixed(1)}`;
        return [
            "character — kinematic capsule controller",
            `drop ${p(ch.drop ?? 0)}  wall ${p(ch.wall ?? 0)}  push ${p(ch.push ?? 0)}`,
            `jump ${p(ch.jumper ?? 0)}  slope30 ${p(ch.slope30 ?? 0)}  slope60 ${p(ch.slope60 ?? 0)}`,
        ].join("\n");
    },
};

// ── probe: minimal floor + Player scene, then the latency measurement (sub-stage 0) ──

async function buildProbe(p: Params): Promise<{ state: State; dispose: () => void }> {
    probeParams = p;
    const { state, dispose } = await run({
        defaults: false,
        capacity: 64,
        plugins: [
            ProfilePlugin,
            SlabPlugin,
            MirrorPlugin,
            TransformsPlugin,
            InputPlugin,
            RenderPlugin,
            PhysicsPlugin,
            CharacterPlugin,
            PlayerPlugin,
            PartPlugin,
            SearPlugin,
            GlazePlugin,
            LoadPlugin,
        ],
    });

    state.add(state.create(), AmbientLight);
    state.add(state.create(), DirectionalLight);
    Physics.step?.configure(cfg);

    // a clear flat floor; top at y = 0.5, so the player rests at y = 1.3.
    staticBox(state, [0, 0, 0], [40, 0.5, 40], [0.28, 0.3, 0.34]);
    const restY = 0.5 + REST_OFFSET;

    // the player: a kinematic capsule the character controller registers + Player drives (look + a
    // first-person follow camera). The camera's Transform IS what we time the input against — orientation
    // through setLook, position through the CPU sweep's pose + the snapshot interpolation, both same-frame.
    const body = state.create();
    state.add(body, Body);
    Body.shape.set(body, ShapeKind.Capsule);
    Body.pos.set(body, 0, restY, 0, 0);
    Body.halfExtents.set(body, 0, HALF_H, 0, RADIUS);
    Body.mass.set(body, 0);
    Body.friction.set(body, 0.8);
    state.add(body, Character);
    Character.maxSlope.set(body, MAX_SLOPE);
    state.add(body, Player);
    playerEid = body;

    const cam = state.create();
    state.add(cam, Transform);
    state.add(cam, Camera);
    state.add(cam, Sear);
    Camera.mode.set(cam, CameraMode.Perspective);
    Player.camera.set(body, cam);
    playerCamEid = cam;

    await frames(3);
    // settle in ENGINE frames (not rAF): grounded, the camera interp inits, the player rests — robust at any
    // load, where a fixed rAF count would be only a few engine frames.
    for (let i = 0; i < 20; i++) await engineStep();

    probeResult = await measureLatency();

    return {
        state,
        dispose() {
            disposeLoad();
            probeParams = null;
            probeResult = null;
            playerEid = -1;
            playerCamEid = -1;
            dispose();
        },
    };
}

// Advance ONE engine frame (Compute.frame increments once per EndFrameSystem). My `frames()` awaits the
// browser's rAF, which under GPU load ticks FASTER than the engine's fence-gated loop — so a raw `frames(1)`
// count is not an engine-frame count, and frame_time read off it would be the rAF cadence, not the engine's.
// Stepping on Compute.frame keeps every probe quantity in engine-frame units, load-robust.
async function engineStep(): Promise<void> {
    const f = Compute.frame;
    for (let i = 0; i < 600; i++) {
        await frames(1);
        if (Compute.frame > f) return;
    }
}

// felt-lag entry: the gate needs ≥1 fixed step (dt 16.7 ms) per render frame so the high-fps fixed-step
// quantization collapses. Calibrate above it with margin (jitter mustn't dip a render frame below 1 step);
// the gate's floor (FELT_MS) is lower, the band between them the calibration's safety margin.
const CALIBRATE_MS = 28; // ≈36 fps — well above 1 step/frame (fixed dt 16.7 ms), the calibration target
// the auto-calibration ramp ceiling — well past the `load` param's manual UI cap (65535), since a 4090-class
// GPU needs ~max to reach felt-lag, so a faster GPU needs headroom (the per-lane iter is a u32 loop bound).
const LOAD_MAX = 262144;

// measure frame_time (ms/engine-frame) over a short window — wall-clock / engine-frames, the fence-gated rate.
async function frameMsOver(window: number): Promise<number> {
    const wall0 = performance.now();
    const f0 = Compute.frame;
    for (let i = 0; i < window; i++) await engineStep();
    const ef = Compute.frame - f0;
    return ef > 0 ? (performance.now() - wall0) / ef : 0;
}

// ramp the GPU-load knob until frame_time reaches the felt-lag regime (per-hardware: a 4090 needs far more
// than a Deck). A manual `load` param pins it instead (for hand exploration). Doubling from a base so a slow
// device converges in a few steps; capped at LOAD_MAX (the auto-ramp ceiling, past the manual param cap).
async function calibrateLoad(): Promise<number> {
    const manual = (probeParams?.load as number | undefined) ?? 0;
    if (manual > 0) {
        setLoad(manual);
        return getLoad();
    }
    let load = 2048;
    setLoad(load);
    let ft = await frameMsOver(16);
    while (ft < CALIBRATE_MS && load < LOAD_MAX) {
        load = Math.min(load * 2, LOAD_MAX);
        setLoad(load);
        ft = await frameMsOver(16);
    }
    return getLoad();
}

async function measureLatency(): Promise<Record<string, number>> {
    const cam = playerCamEid;
    const body = playerEid;
    const out: Record<string, number> = {};

    // 1) calibrate the GPU-load knob into the felt-lag regime, so the fixed-step quantization that dominates
    //    input latency at high fps collapses and the pose-vs-orientation comparison is apples-to-apples.
    out.load = await calibrateLoad();

    // 2) frame_time at the calibrated load (engine-frame units: wall-clock / engine-frames, NOT per rAF tick).
    const frameMs = await frameMsOver(60);
    out.frameMs = frameMs;
    out.fps = frameMs > 0 ? 1000 / frameMs : 0;
    out.fenceInFlight = Compute.pending();

    // 3) orientation latency — setLook off Player.yaw in the sim group, CPU same-frame: the floor reference.
    out.orientFrames = await injectOrientation(cam, body);
    out.orientMs = out.orientFrames * frameMs;

    // 4) the position CONTROLLER latency (THE GATE): input → pose(eid), the CPU sweep's CharState written
    //    this fixed tick, BEFORE any GPU work. Same-frame (≈ orientation) ⇒ no fence/Mirror in the path. The
    //    raw pose has no interpolation confound — the cleanest read of the part the CPU sweep fixes.
    out.poseFrames = await injectMove(body, () => (pose(body, _probePose) ? _probePose[0] : null));
    out.poseMs = out.poseFrames * frameMs;

    // 5) the full input → camera.pos (pose + the kept one-tick fixed-step interpolation) — a reporter, since
    //    the interpolation tick is a deliberate, frame_time-independent lag (Gaffer), not a readback.
    out.camFrames = await injectMove(body, () => Transform.pos.x.get(cam));
    out.camMs = out.camFrames * frameMs;

    out.deltaFrames = out.poseFrames - out.orientFrames;
    return out;
}

const PROBE_REPS = 4;

// rotate Player.yaw by a step; count engine frames until the camera Transform.rot reflects it.
// PlayerControlSystem (simulation) setLooks off Player.yaw every frame, so the change lands the engine
// frame after injection — the ~1-frame floor, no readback.
async function injectOrientation(cam: number, body: number): Promise<number> {
    const Cap = 30;
    const Eps = 1e-3;
    let sum = 0;
    let count = 0;
    for (let rep = 0; rep < PROBE_REPS; rep++) {
        const base = Player.yaw.get(body);
        const r0 = [
            Transform.rot.x.get(cam),
            Transform.rot.y.get(cam),
            Transform.rot.z.get(cam),
            Transform.rot.w.get(cam),
        ];
        const f0 = Compute.frame;
        Player.yaw.set(body, base + 0.5);
        let delta = Cap;
        for (let i = 0; i < Cap; i++) {
            await engineStep();
            const d =
                Math.abs(Transform.rot.x.get(cam) - r0[0]) +
                Math.abs(Transform.rot.y.get(cam) - r0[1]) +
                Math.abs(Transform.rot.z.get(cam) - r0[2]) +
                Math.abs(Transform.rot.w.get(cam) - r0[3]);
            if (d > Eps) {
                delta = Compute.frame - f0;
                break;
            }
        }
        sum += delta;
        count++;
        Player.yaw.set(body, base);
        for (let i = 0; i < 8; i++) await engineStep(); // settle in engine frames before the next rep
    }
    return count ? sum / count : 0;
}

// hold a forward move; count engine frames until `readX` (the x-coordinate of the thing under test) moves.
// Under felt-lag (≥1 fixed step / render frame) a CPU-same-frame reader lands in 1 engine frame; a reader
// gated on a GPU readback would push to 2–3. `readX` returns null when unavailable (the char isn't registered
// yet) → that rep is skipped. Averages over PROBE_REPS, settling the char to rest between reps.
async function injectMove(body: number, readX: () => number | null): Promise<number> {
    const Cap = 40;
    const Eps = 5e-3;
    let sum = 0;
    let count = 0;
    for (let rep = 0; rep < PROBE_REPS; rep++) {
        const x0 = readX();
        if (x0 === null) break;
        const f0 = Compute.frame;
        let delta = Cap;
        for (let i = 0; i < Cap; i++) {
            move(body, 3, 0);
            await engineStep();
            const x = readX();
            if (x !== null && Math.abs(x - x0) > Eps) {
                delta = Compute.frame - f0;
                break;
            }
        }
        sum += delta;
        count++;
        for (let i = 0; i < 20; i++) {
            move(body, 0, 0);
            await engineStep(); // settle back in engine frames
        }
    }
    return count ? sum / count : 0;
}

// the gate's felt-lag floor: at this frame_time every render frame runs ≥1 fixed step, so the sweep updates
// pose on the injection frame (poseFrames = 1) — below it, high-fps quantization inflates poseFrames and the
// comparison is meaningless. The pose latency is integer ~1 (CPU same-frame, no interpolation jitter), so a
// re-introduced readback (≥1 fence-gated frame) shows as Δ ≥ 2; TOL = 1 separates them robustly.
const FELT_MS = 22;
const POSE_TOL = 1;

function probeChecks(): Check[] {
    const r = probeResult;
    if (!r) return [{ name: "probe", pass: false, detail: "no probe result" }];
    const f2 = (x: number): string => x.toFixed(2);
    const inFeltLag = r.frameMs >= FELT_MS;
    return [
        {
            name: "probe: frame_time (the GPU-load knob, calibrated into the felt-lag regime)",
            pass: true,
            detail: `${f2(r.frameMs)} ms/frame (${f2(r.fps)} fps) at load=${r.load | 0} iters/lane; fence pending ${r.fenceInFlight} (loop runs ≤2 in flight)`,
            data: r,
        },
        {
            name: "probe: orientation latency (setLook, CPU same-frame) — the floor",
            pass: true,
            detail: `${f2(r.orientFrames)} frames → ${f2(r.orientMs)} ms`,
            data: r,
        },
        {
            name: "probe: input → camera.pos latency (pose + the kept one-tick interpolation)",
            pass: true,
            detail: `${f2(r.camFrames)} frames → ${f2(r.camMs)} ms (= pose + the frame_time-independent interp tick)`,
            data: r,
        },
        {
            // the regression: the position controller's pose updates same-frame as orientation — no GPU fence
            // or Mirror in the path, so its latency holds at the floor under load (a readback would be ×frame_time).
            name: "position input→camera carries no GPU readback (pose is CPU same-frame as orientation)",
            pass: inFeltLag && r.deltaFrames <= POSE_TOL,
            detail: inFeltLag
                ? `under load (${f2(r.frameMs)} ms/frame): pose ${f2(r.poseFrames)} vs orientation ${f2(r.orientFrames)} frames — Δ ${f2(r.deltaFrames)} ≤ ${POSE_TOL}; a readback would add ≥1 fence-gated frame (×${f2(r.frameMs)} ms)`
                : `frame_time ${f2(r.frameMs)} ms < ${FELT_MS} ms felt-lag floor — the load knob (max ${LOAD_MAX}) couldn't slow this GPU enough to validate; pose ${f2(r.poseFrames)} vs orientation ${f2(r.orientFrames)}`,
            data: r,
        },
    ];
}

// ── live behavioral gates (read the settled GPU state through Mirror, grounded via character/core) ──

async function liveGates(): Promise<Check[]> {
    if (!bodyMirror) return [{ name: "character live", pass: false, detail: "no physics step" }];
    await settle(bodyMirror);
    const snap = bodyMirror.snapshot;
    if (!snap) return [{ name: "character live", pass: false, detail: "no snapshot" }];
    const s = new Float32Array(snap.bytes);
    const cap = s.length / (BODY_VEC4 * 4);
    const pos = (eid: number): Vec3 => [
        s[(0 * cap + eid) * 4],
        s[(0 * cap + eid) * 4 + 1],
        s[(0 * cap + eid) * 4 + 2],
    ];
    const restY = 0.5 + REST_OFFSET;
    const checks: Check[] = [];

    const dy = pos(ch.drop)[1];
    checks.push({
        name: "drop-to-rest (settles on the floor, grounded)",
        pass: Math.abs(dy - restY) < 0.05 && grounded(ch.drop),
        detail: `y ${dy.toFixed(3)} (rest ${restY}), grounded ${grounded(ch.drop)}`,
    });

    // the sweep stops the char before the wall interior (x < -2, never tunnels in) and presses up near the
    // surface (x > -2.7, not stuck far back). The exact press-in depth depends on how long it's been walking,
    // so the band is tolerant either side of the geometric surface (-2.30) — the no-tunnel invariant is x < -2.
    const wx = pos(ch.wall);
    checks.push({
        name: "wall-stop (the sweep halts the character at a static wall)",
        pass: wx[0] < -2 && wx[0] > -2.7 && Math.abs(wx[1] - restY) < 0.1,
        detail: `char x ${wx[0].toFixed(3)} (face -2, surface ~${(-2 - RADIUS).toFixed(2)}), y ${wx[1].toFixed(2)}`,
    });

    const wb = pos(wallBoxEid);
    checks.push({
        name: "kin-static bounded (box behind the wall character is undisturbed)",
        pass: wb.every(Number.isFinite) && Math.abs(wb[0] - -9) < 0.2 && wb[1] > 0,
        detail: `box ${wb.map((v) => v.toFixed(2)).join(", ")} (start -9)`,
    });

    const pb = pos(pushBoxEid);
    const pcx = pos(ch.push)[0];
    checks.push({
        name: "character shoves a dynamic box at full speed without overtaking it",
        pass: pb.every(Number.isFinite) && pb[0] > 1 && pcx < pb[0],
        detail: `box x ${pb[0].toFixed(3)} (start -1), char x ${pcx.toFixed(3)} (behind = no overtake) | ${pushTraj}`,
    });

    const pinPos = pos(pinBoxEid);
    const pwx = pos(ch.pushWall);
    checks.push({
        name: "push-into-wall stays bounded, no explosion (the legacy explosive case)",
        pass:
            pinPos.every(Number.isFinite) &&
            pwx.every(Number.isFinite) &&
            Math.hypot(pinPos[0], pinPos[1], pinPos[2]) < 30 &&
            pwx[0] < 3.5,
        detail: `box ${pinPos.map((v) => v.toFixed(2)).join(", ")} (wall face 3.5), char x ${pwx[0].toFixed(2)}`,
    });

    // 30° (walkable) drops onto the ramp and holds; 60° (too steep) slides off it. Measure total horizontal
    // displacement from the build-time spawn x against the settled final pose only — no mid-window sample, so
    // the verdict is independent of how much sim-time elapses per frames(30) (the headless rAF rate varies by
    // platform). 30° lands at its spawn x and holds (~0.01 m); 60° slides ~2.2 m off the ramp to the floor.
    const s30 = Math.abs(pos(ch.slope30)[0] - slope30SpawnX);
    const s60 = Math.abs(pos(ch.slope60)[0] - slope60SpawnX);
    checks.push({
        name: "slope: 30° holds (walkable), 60° slides (too steep)",
        pass: grounded(ch.slope30) && s30 < 0.8 && s60 > 1.5,
        detail: `30° slid ${s30.toFixed(3)} m from spawn (grounded ${grounded(ch.slope30)}), 60° slid ${s60.toFixed(3)} m (final y ${pos(ch.slope60)[1].toFixed(2)})`,
    });

    return checks;
}

// ── perf reporter ────────────────────────────────────────────────────────────

const STEP_PASSES = [
    "phys:pack",
    "phys:aabb",
    "bvh:sort",
    "bvh:build",
    "phys:broadphase",
    "phys:collide",
    "phys:csr",
    "phys:coloring",
    "phys:inertial",
    "phys:primal",
    "phys:dual",
    "phys:velocity",
    "phys:compose",
] as const;

// always-pass perf REPORTER (not a correctness gate): the per-step solver spans as structured `Check.data`,
// the seam scripts/physics-bench.ts reads. The character sweep is CPU (no GPU span) — its cost isn't here.
async function measured(): Promise<Check> {
    const get = (name: string): number => Profile.gpu.get(name) ?? 0;
    const data: Record<string, number> = {};
    for (const n of STEP_PASSES) data[n] = get(n);
    const full = STEP_PASSES.reduce((sum, n) => sum + data[n], 0);
    data.dispatchedColors = Physics.step?.dispatchedColors ?? 0;
    data.bytes = Physics.step?.bytes ?? 0;
    const resolved = get("phys:primal") > 0;
    const label = (n: string): string => n.replace("phys:", "").replace("bvh:", "bvh.");
    const parts = STEP_PASSES.map((n) => `${label(n)} ${data[n].toFixed(3)}`).join(" · ");
    return {
        name: "measured (solver spans)",
        pass: true,
        detail: resolved ? `step ~${full.toFixed(3)} ms = ${parts} ms` : "no solver spans resolved",
        data,
    };
}

register(scenario);
