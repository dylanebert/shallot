import {
    Body,
    Color,
    compose,
    invert,
    loadGltf,
    Physics,
    type Plugin,
    placeGltf,
    ShapeKind,
    type State,
    type System,
    Transform,
    Tumble,
} from "@dylanebert/shallot";
import { LiveSkin, skinMatrix } from "@dylanebert/shallot/gltf/core";
import { qRotate } from "@dylanebert/shallot/physics/core";

// a ragdoll is a physics skeleton wearing a skinned mesh: capsule bodies fall and tangle under the
// solver, and a live joint palette copies their poses onto the character's vertices every frame, so the
// mesh crumples with them. The bones are ordinary `Body` entities (physics, picking, and the character
// sweep all see them); the joints between them ride the tumble backend's world directly, past the
// substrate's `Spring`/`Joint` mapping, for the cone/twist/hinge limits a ragdoll needs. The rig itself is
// imported in code below, because a live-skinned import is a programmatic call, not a scene mesh reference.
// The cone/twist joints have no published substrate-surface equivalent yet, so the joints ride the escape
// hatch; the `LiveSkin` palette is the published half. The gym twin `ragdoll-ragdoll` is the oracle-gated gold.

type V3 = [number, number, number];
type Q4 = [number, number, number, number];

// quaternions are [x, y, z, w]; rotation reuses physics/core's `qRotate`
const qMul = (a: Q4, b: Q4): Q4 => [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
];
const qConj = (q: Q4): Q4 => [-q[0], -q[1], -q[2], q[3]];
const qRotA = (q: Q4, v: V3): V3 => qRotate(q[0], q[1], q[2], q[3], v[0], v[1], v[2]);

// the shortest-arc quaternion taking unit `a` onto unit `b`; antiparallel picks any perpendicular axis
function qFromTo(a: V3, b: V3): Q4 {
    const d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    if (d > 1 - 1e-8) return [0, 0, 0, 1];
    if (d < -1 + 1e-8) {
        const ax: V3 = Math.abs(a[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
        const p: V3 = [
            a[1] * ax[2] - a[2] * ax[1],
            a[2] * ax[0] - a[0] * ax[2],
            a[0] * ax[1] - a[1] * ax[0],
        ];
        const l = Math.hypot(p[0], p[1], p[2]);
        return [p[0] / l, p[1] / l, p[2] / l, 0];
    }
    const c: V3 = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const w = 1 + d;
    const l = Math.hypot(c[0], c[1], c[2], w);
    return [c[0] / l, c[1] / l, c[2] / l, w / l];
}

// shortest-arc nlerp for render interpolation between fixed ticks (flip `a` into `b`'s hemisphere, lerp,
// renormalize) — the same blend the engine's transform compose uses
function nlerp(a: Q4, b: Q4, t: number): Q4 {
    const dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    const f = dot < 0 ? -1 : 1;
    const q: Q4 = [
        a[0] * f * (1 - t) + b[0] * t,
        a[1] * f * (1 - t) + b[1] * t,
        a[2] * f * (1 - t) + b[2] * t,
        a[3] * f * (1 - t) + b[3] * t,
    ];
    const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
    return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}

// three tables. `BONES` are 11 capsules in the rig's object space (glTF is Z-up, feet at z = 0), placed
// to RiggedFigure's skeleton; the order is fixed because the solver keys determinism on creation order.
// `MAP` says which glTF joints each bone drives, and `JOINTS` links the bones with limited joints: a
// `ball` (cone + twist) at the spine, neck, shoulders, and hips; a `hinge` at the elbows and knees; one
// `filter` so the thighs don't collide with each other
const BONES: { name: string; a: V3; b: V3; r: number; mass: number }[] = [
    { name: "pelvis", a: [-0.09, 0, 0.66], b: [0.09, 0, 0.66], r: 0.09, mass: 2.5 },
    { name: "chest", a: [0, 0, 0.78], b: [0, 0, 1.04], r: 0.11, mass: 3.5 },
    { name: "head", a: [0, 0, 1.21], b: [0, 0, 1.33], r: 0.09, mass: 1 },
    { name: "upperArmL", a: [0.088, 0.01, 1.074], b: [0.306, 0.023, 0.964], r: 0.05, mass: 0.5 },
    { name: "upperArmR", a: [-0.088, 0.01, 1.074], b: [-0.306, 0.023, 0.964], r: 0.05, mass: 0.5 },
    { name: "lowerArmL", a: [0.306, 0.023, 0.964], b: [0.447, -0.065, 0.882], r: 0.045, mass: 0.4 },
    {
        name: "lowerArmR",
        a: [-0.306, 0.023, 0.964],
        b: [-0.447, -0.065, 0.882],
        r: 0.045,
        mass: 0.4,
    },
    { name: "thighL", a: [0.068, -0.001, 0.614], b: [0.077, -0.058, 0.354], r: 0.055, mass: 1.4 },
    { name: "thighR", a: [-0.068, -0.001, 0.614], b: [-0.077, -0.058, 0.354], r: 0.055, mass: 1.4 },
    { name: "calfL", a: [0.077, -0.058, 0.354], b: [0.078, 0.002, 0.085], r: 0.05, mass: 0.9 },
    { name: "calfR", a: [-0.077, -0.058, 0.354], b: [-0.078, 0.002, 0.085], r: 0.05, mass: 0.9 },
];

const MAP: Record<string, string[]> = {
    pelvis: ["torso_joint_1"],
    chest: ["torso_joint_2", "torso_joint_3"],
    head: ["neck_joint_1", "neck_joint_2"],
    upperArmL: ["arm_joint_L_1"],
    upperArmR: ["arm_joint_R_1"],
    lowerArmL: ["arm_joint_L_2", "arm_joint_L_3"],
    lowerArmR: ["arm_joint_R_2", "arm_joint_R_3"],
    thighL: ["leg_joint_L_1"],
    thighR: ["leg_joint_R_1"],
    calfL: ["leg_joint_L_2", "leg_joint_L_3", "leg_joint_L_5"],
    calfR: ["leg_joint_R_2", "leg_joint_R_3", "leg_joint_R_5"],
};

// `axis` is the joint frame's Z in object space — tumble reads it as the cone axis (ball) and hinge axis
// (hinge). Shoulder cones point down the arm, hips point down, spine and neck point up.
const JOINTS: { kind: "ball" | "hinge" | "filter"; a: string; b: string; pivot?: V3; axis?: V3 }[] =
    [
        { kind: "ball", a: "pelvis", b: "chest", pivot: [0, 0, 0.72], axis: [0, 0, 1] },
        { kind: "ball", a: "chest", b: "head", pivot: [0, 0, 1.13], axis: [0, 0, 1] },
        {
            kind: "ball",
            a: "chest",
            b: "upperArmL",
            pivot: [0.088, 0.01, 1.074],
            axis: [0.218, 0.013, -0.11],
        },
        {
            kind: "ball",
            a: "chest",
            b: "upperArmR",
            pivot: [-0.088, 0.01, 1.074],
            axis: [-0.218, 0.013, -0.11],
        },
        {
            kind: "hinge",
            a: "upperArmL",
            b: "lowerArmL",
            pivot: [0.306, 0.023, 0.964],
            axis: [0, 1, 0],
        },
        {
            kind: "hinge",
            a: "upperArmR",
            b: "lowerArmR",
            pivot: [-0.306, 0.023, 0.964],
            axis: [0, -1, 0],
        },
        { kind: "ball", a: "pelvis", b: "thighL", pivot: [0.068, -0.001, 0.614], axis: [0, 0, -1] },
        {
            kind: "ball",
            a: "pelvis",
            b: "thighR",
            pivot: [-0.068, -0.001, 0.614],
            axis: [0, 0, -1],
        },
        { kind: "hinge", a: "thighL", b: "calfL", pivot: [0.077, -0.058, 0.354], axis: [1, 0, 0] },
        { kind: "hinge", a: "thighR", b: "calfR", pivot: [-0.077, -0.058, 0.354], axis: [1, 0, 0] },
        { kind: "filter", a: "thighL", b: "thighR" },
    ];

const RIG = "rig/RiggedFigure.gltf";
const DROP = 1.0; // feet above the floor at spawn
const TILT = 0.25; // a lean about world Z so the topple is deterministic, not knife-edge

interface Bone {
    eid: number;
    invBind: Float32Array; // the bone's object-space bind inverse — skinMatrix's right factor
    joints: number[]; // palette indices this bone drives
    spawnPos: V3;
    spawnQuat: Q4;
}

// runtime state, reset each build (a State is rebuilt on every scene switch or play/stop — ecs.md)
let figure = -1;
let bones: Bone[] = [];
let bindPos: V3 = [0, 0, 0]; // the pelvis bone's object-space bind — the instance transform's fixed factor
let bindQuat: Q4 = [0, 0, 0, 1];
let r0: Q4 = [0, 0, 0, 1]; // spawn orientation: object Z-up → world Y-up, plus the lean
let wired = 0;
let palette = new Float32Array(0);
let prev = new Float32Array(0); // per-bone fixed-tick poses (7 lanes: pos xyz + quat xyzw), the lerp pair
let curr = new Float32Array(0);
let tick = -1;
let sampled = false;

// import the rig on the `live` path (a runtime palette, not baked clips), place it, then create the 11
// capsule bodies. Each body's inverse bind is `invert(compose(...))` of where the bone rests in object
// space, the factor `skinMatrix` needs to turn a live pose into a palette entry. The pelvis is the
// palette's root, so its bind is remembered for the instance transform
async function build(state: State): Promise<void> {
    bones = [];
    wired = 0;
    tick = -1;
    sampled = false;

    const asset = await loadGltf(state, RIG, { live: true });
    const handle = asset.meshes.find((m) => m.live);
    if (!handle) throw new Error("[ragdoll] RiggedFigure did not import live");
    figure = placeGltf(state, handle);
    Color.rgba.set(figure, 0.95, 0.7, 0.35, 1); // untextured, so the tint is the whole look
    palette = new Float32Array(16 * handle.jointCount);
    prev = new Float32Array(7 * BONES.length);
    curr = new Float32Array(7 * BONES.length);

    // the palette is ordered by the skin's joint list, so resolve the bone map's names to their indices
    const json = await (await fetch(RIG)).json();
    const names: string[] = json.skins[0].joints.map((n: number) => json.nodes[n].name);
    const jointIdx = new Map(names.map((n, i) => [n, i] as const));

    // spawn frame: object Z-up → world Y-up (a −90° x-rotation), leaned TILT about world Z, feet DROP up
    const S = Math.SQRT1_2;
    r0 = qMul([0, 0, Math.sin(TILT / 2), Math.cos(TILT / 2)], [-S, 0, 0, S]);
    const origin: V3 = [0, DROP, 0];
    for (const bone of BONES) {
        const mid: V3 = [
            (bone.a[0] + bone.b[0]) / 2,
            (bone.a[1] + bone.b[1]) / 2,
            (bone.a[2] + bone.b[2]) / 2,
        ];
        const dir: V3 = [bone.b[0] - bone.a[0], bone.b[1] - bone.a[1], bone.b[2] - bone.a[2]];
        const len = Math.hypot(dir[0], dir[1], dir[2]);
        // a capsule's segment runs along local +Y, so align that to the bone direction
        const align = qFromTo([0, 1, 0], [dir[0] / len, dir[1] / len, dir[2] / len]);
        const wr = qRotA(r0, mid);
        const spawnPos: V3 = [origin[0] + wr[0], origin[1] + wr[1], origin[2] + wr[2]];
        const spawnQuat = qMul(r0, align);

        const eid = state.create();
        state.add(eid, Body);
        Body.shape.set(eid, ShapeKind.Capsule);
        Body.pos.set(eid, spawnPos[0], spawnPos[1], spawnPos[2], 0);
        Body.quat.set(eid, spawnQuat[0], spawnQuat[1], spawnQuat[2], spawnQuat[3]);
        Body.halfExtents.set(eid, 0, len / 2, 0, bone.r);
        Body.mass.set(eid, bone.mass);

        const joints = MAP[bone.name].map((n) => {
            const j = jointIdx.get(n);
            if (j === undefined) throw new Error(`[ragdoll] rig has no joint named ${n}`);
            return j;
        });
        const objBind = compose(
            mid[0],
            mid[1],
            mid[2],
            align[0],
            align[1],
            align[2],
            align[3],
            1,
            1,
            1,
        );
        bones.push({
            eid,
            invBind: invert(objBind, new Float32Array(16)),
            joints,
            spawnPos,
            spawnQuat,
        });
        if (bone.name === "pelvis") {
            bindPos = mid;
            bindQuat = align;
        }
    }

    state.addSystem(driver);
}

// the bodies only marshal into the tumble world on the first fixed tick, so wiring waits until every
// `Tumble.body(eid)` resolves. Local anchor frames come from the spawn pose analytically, because the
// bodies have already stepped by wire time, so asking the live world for a local point would fold the
// first free-fall ticks into the joint. Cone, twist, and hinge limits keep the tangle human; the motors
// let it settle rather than flail
function wire(): void {
    const world = Tumble.world;
    if (!world) return;
    const handles = bones.map((b) => Tumble.body(b.eid));
    if (handles.some((h) => !h)) return;

    const index = new Map(BONES.map((b, i) => [b.name, i]));
    const origin: V3 = [0, DROP, 0];
    for (const j of JOINTS) {
        const ia = index.get(j.a)!;
        const ib = index.get(j.b)!;
        const A = handles[ia]!;
        const B = handles[ib]!;
        if (j.kind === "filter") {
            world.createFilterJoint(A, B);
            wired++;
            continue;
        }
        const al = Math.hypot(j.axis![0], j.axis![1], j.axis![2]);
        const axisW = qRotA(r0, [j.axis![0] / al, j.axis![1] / al, j.axis![2] / al]);
        const qJ = qFromTo([0, 0, 1], axisW); // both frames share this world orientation → rest rotation identity
        const pivotW: V3 = [0, 0, 0];
        const rot = qRotA(r0, j.pivot!);
        for (let k = 0; k < 3; k++) pivotW[k] = origin[k] + rot[k];
        const frame = (i: number) => {
            const b = bones[i];
            const local = qRotA(qConj(b.spawnQuat), [
                pivotW[0] - b.spawnPos[0],
                pivotW[1] - b.spawnPos[1],
                pivotW[2] - b.spawnPos[2],
            ]);
            const q = qMul(qConj(b.spawnQuat), qJ);
            return {
                p: { x: local[0], y: local[1], z: local[2] },
                q: { v: { x: q[0], y: q[1], z: q[2] }, s: q[3] },
            };
        };
        if (j.kind === "ball") {
            world.createSphericalJoint(A, B, {
                localFrameA: frame(ia),
                localFrameB: frame(ib),
                enableConeLimit: true,
                coneAngle: 0.9,
                enableTwistLimit: true,
                lowerTwistAngle: -0.4,
                upperTwistAngle: 0.4,
                enableMotor: true,
                maxMotorTorque: 1.5,
                motorVelocity: { x: 0, y: 0, z: 0 },
            });
        } else {
            world.createRevoluteJoint(A, B, {
                localFrameA: frame(ia),
                localFrameB: frame(ib),
                enableLimit: true,
                lowerAngle: -0.1,
                upperAngle: 2.2,
                enableMotor: true,
                maxMotorTorque: 1.5,
                motorSpeed: 0,
            });
        }
        wired++;
    }
}

// each fixed tick, sample every bone's pose; each render frame, interpolate at `fixedAlpha` for
// smoothness, track the instance transform to the pelvis (so frustum cull and picking keep a real root
// while the palette stays small), then write each joint's palette entry as its bone's rigid delta from
// bind via `skinMatrix(relPose, boneInverseBind)`. The skin surface blends the palette into the mesh in
// the same vertex stage that feeds the shadow pass, so the shadow crumples for free
const driver: System = {
    name: "ragdoll-driver",
    group: "simulation",
    update(state: State) {
        if (figure < 0 || bones.length === 0) return;
        if (wired === 0) wire();
        const backend = Physics.backend;
        if (!backend) return;

        if (state.time.fixedTick !== tick) {
            tick = state.time.fixedTick;
            for (let i = 0; i < bones.length; i++) {
                const s = backend.readBody(bones[i].eid);
                if (!s) return; // not marshaled yet — hold the seeded bind pose
                const o = i * 7;
                if (sampled) prev.set(curr.subarray(o, o + 7), o);
                curr.set(
                    [s.pos[0], s.pos[1], s.pos[2], s.quat[0], s.quat[1], s.quat[2], s.quat[3]],
                    o,
                );
                if (!sampled) prev.set(curr.subarray(o, o + 7), o);
            }
            sampled = true;
        }
        if (!sampled) return;

        const t = state.time.fixedAlpha;
        const pose = (i: number): { p: V3; q: Q4 } => {
            const o = i * 7;
            return {
                p: [
                    prev[o] + (curr[o] - prev[o]) * t,
                    prev[o + 1] + (curr[o + 1] - prev[o + 1]) * t,
                    prev[o + 2] + (curr[o + 2] - prev[o + 2]) * t,
                ],
                q: nlerp(
                    [prev[o + 3], prev[o + 4], prev[o + 5], prev[o + 6]],
                    [curr[o + 3], curr[o + 4], curr[o + 5], curr[o + 6]],
                    t,
                ),
            };
        };

        const pelvis = pose(0);
        // instance transform = pelvisNow · pelvisBind⁻¹, so the entity rides the pelvis
        const instQ = qMul(pelvis.q, qConj(bindQuat));
        const off = qRotA(instQ, bindPos);
        Transform.pos.set(
            figure,
            pelvis.p[0] - off[0],
            pelvis.p[1] - off[1],
            pelvis.p[2] - off[2],
            0,
        );
        Transform.rot.set(figure, instQ[0], instQ[1], instQ[2], instQ[3]);

        // each bone's pose expressed relative to the pelvis root, then into a palette entry per joint
        const qTi = qMul(bindQuat, qConj(pelvis.q));
        for (let i = 0; i < bones.length; i++) {
            const bone = bones[i];
            const b = pose(i);
            const d = qRotA(qTi, [
                b.p[0] - pelvis.p[0],
                b.p[1] - pelvis.p[1],
                b.p[2] - pelvis.p[2],
            ]);
            const relP: V3 = [bindPos[0] + d[0], bindPos[1] + d[1], bindPos[2] + d[2]];
            const relQ = qMul(qTi, b.q);
            for (const j of bone.joints) {
                skinMatrix(relP, relQ, bone.invBind, palette.subarray(j * 16, j * 16 + 16));
            }
        }
        LiveSkin.writePalette(figure, palette);
    },
};

export const Ragdoll = {
    name: "Ragdoll",
    warm: build,
} satisfies Plugin;

export default Ragdoll;
