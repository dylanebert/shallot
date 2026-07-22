// Bit-exact fixture gate — the stage-8 regression contract. Each scene is rebuilt through the
// public API, stepped, and its FNV-1a world-state hash asserted equal to the C reference's, per
// step. The fixtures come from reference/box3d (branch `harness`) built scalar + force-overflow;
// regenerate with `bun run gen-fixtures`. On divergence the first mismatched step is reported with
// the port's body states so the drift can be localized against the fixture's periodic dumps.
//
// The scene builders mirror fixtures/gen.c exactly (same creation order, same params). Sleep is
// per-scene, matching the generator: stage-8 scenes run the awake path with sleep off; stage-9 scenes
// (sphere-sleep / box-sleep / wake-drop) turn it on and gate the sleep-step index bit-for-bit.

import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { BodyFlags, getBodySim, getBodyState } from "./body";
import { B_FLAGS, B_STATE, IDENT_RECORDS, N_BODY } from "./bodycolumns";
import { STATE_LIVE, STATE_STRIDE } from "./columns";
import { hashWorldState } from "./hash";
import {
    type Body,
    BodyType,
    type CompoundData,
    createCompound,
    createCylinder,
    createGrid,
    createGridMesh,
    createHull,
    createMesh,
    createRock,
    createTorusMesh,
    defaultFilter,
    defaultSurfaceMaterial,
    type HullData,
    makeBoxHull,
    makeOffsetBoxHull,
    type Vec3,
    World,
} from "./index";
import { init, kernel, sharedBytes, shutdown, threads } from "./kernel";
import { computeCosSin, DEG_TO_RAD, offsetPos, quat, vec3 } from "./math";

const QUAT_ID = { v: { x: 0, y: 0, z: 0 }, s: 1 };

/**
 * Every thread's identity record must still be the identity after a solve: zero velocity + delta,
 * identity delta-rotation, DYNAMIC.
 *
 * The wide gather remaps null/static lanes onto the *running worker's* record and its whole-vector
 * scatter writes that record back. A shared record would be a same-address concurrent write across
 * blocks — benign by value, but a data race, and the one element of the state column that isn't
 * write-disjoint. This asserts the per-worker records exist and stay pristine: a record polluted by a
 * real body's velocity would silently poison every later null-lane gather.
 */
function expectIdentityRecords(): void {
    const k = kernel();
    const cap = k.bodyCap();
    if (cap === 0) return;
    const buf = k.memory.buffer;
    const layout = new Uint32Array(buf, k.bodyLayoutPtr(), N_BODY);
    const state = new Float32Array(
        buf,
        layout[B_STATE] + cap * STATE_STRIDE * 4,
        IDENT_RECORDS * STATE_STRIDE,
    );
    const flags = new Uint32Array(buf, layout[B_FLAGS] + cap * 4, IDENT_RECORDS);
    const dynamic = BodyFlags.dynamicFlag;
    for (let w = 0; w < IDENT_RECORDS; ++w) {
        const o = w * STATE_STRIDE;
        for (let i = 0; i < STATE_LIVE; ++i) {
            // Slot 12 is deltaRotation.s (identity); the rest of the live record is zero.
            expect([w, i, state[o + i]]).toEqual([w, i, i === 12 ? 1 : 0]);
        }
        expect(flags[w] & dynamic).toBe(dynamic);
    }
}

const fround = Math.fround;

type BodyDump = { p: number[]; q: number[]; v?: number[]; w?: number[] };
type Fixture = {
    scene: string;
    timeStep: number;
    subStepCount: number;
    stepCount: number;
    gravity: number[];
    hashes: string[];
    states: { step: number; bodies: BodyDump[] }[];
    // Present only for scenes whose static ground the port cannot rebuild bit-exactly (bench-trees'
    // libm-sinf wave mesh): the C ground vertices the port feeds to createMesh.
    groundVertices?: number[][];
};

function loadFixture(scene: string): Fixture {
    // Fixtures live at the package's oracle tier (packages/shallot/tests/tumble/fixtures/), outside
    // src/ so npm's files:["src"] never ships them — engine dir is src/standard/tumble/engine.
    const path = resolve(import.meta.dir, "../../../../tests/tumble/fixtures", `${scene}.json`);
    return JSON.parse(readFileSync(path, "utf8")) as Fixture;
}

function toHex(h: bigint): string {
    return `0x${h.toString(16).padStart(16, "0")}`;
}

// Every live body's transform + velocity, in the same id order b3HashWorldState walks — mirrors the
// generator's state dump so a divergence can be read against the fixture.
function dumpBodies(world: World): BodyDump[] {
    const out: BodyDump[] = [];
    const state = world.state;
    for (let i = 0; i < state.bodies.length; ++i) {
        const body = state.bodies[i];
        if (body.id !== i) {
            continue;
        }
        const sim = getBodySim(state, body);
        const dump: BodyDump = {
            p: [sim.transform.p.x, sim.transform.p.y, sim.transform.p.z],
            q: [sim.transform.q.v.x, sim.transform.q.v.y, sim.transform.q.v.z, sim.transform.q.s],
        };
        const bs = getBodyState(state, body);
        if (bs !== null) {
            dump.v = [bs.linearVelocity.x, bs.linearVelocity.y, bs.linearVelocity.z];
            dump.w = [bs.angularVelocity.x, bs.angularVelocity.y, bs.angularVelocity.z];
        }
        out.push(dump);
    }
    return out;
}

function createGround(world: World, halfExtent: number): void {
    const body = world.createBody({ position: { x: 0, y: -1, z: 0 } });
    body.createHull({}, makeBoxHull(halfExtent, 1.0, halfExtent));
}

// A static grid-mesh floor in the xz-plane at y = 0 (8x8 unit cells centered on the origin).
function createMeshFloor(world: World): void {
    const mesh = createGridMesh(8, 8, 1.0, 0, true);
    const body = world.createBody({ position: { x: 0, y: 0, z: 0 } });
    body.createMesh({}, mesh, { x: 1, y: 1, z: 1 });
}

// A static flat 8x8 grid height field, offset to centre on the origin (x,z span [-3.5, 3.5]).
function createHeightFieldFloor(world: World): void {
    const hf = createGrid(8, 8, { x: 1, y: 1, z: 1 }, false);
    const body = world.createBody({ position: { x: -3.5, y: 0, z: -3.5 } });
    body.createHeightField({}, hf);
}

const PI = fround(Math.PI);
// A joint frame at a local pivot with identity rotation.
function frame(x: number, y: number, z: number) {
    return { p: { x, y, z }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };
}

// bench-large-world scene constants (mirror fixtures/gen.c's BENCH_LW_* defines).
const BENCH_LW_CELL = 10.0;
const BENCH_LW_GRID = 12;
const BENCH_LW_SPHERES = 12;
const BENCH_LW_INTERVAL = 5;

// bench-trees reduced-scale constants (mirror fixtures/gen.c's TREES_* defines).
const TREES_XCOUNT = 24;
const TREES_ZCOUNT = 32;
const TREES_BODIES = 3;

// bench-junkyard reduced-scale constants (mirror fixtures/gen.c's JUNK_* defines).
const JUNK_RADIUS = 35.0;
const JUNK_ROCK_BASE = -4.0;
const JUNK_ROCK_SPACING = 4.0;
const JUNK_ROCK_COUNT = 3;
const JUNK_ROCK_HEIGHT = 2.0;

// bench-rain reduced-scale constants (mirror fixtures/gen.c's RAINF_* defines).
const RAINF_GROUND_HALF = 6;
const RAINF_CELL = 1.5;
const RAINF_TORUS_MAJOR = 3.0;
const RAINF_HUMAN_COUNT = 3;
const RAINF_SPAWN_INTERVAL = 40;

const f = fround;
type V3 = [number, number, number];
type Q4 = [number, number, number, number];

// The ragdoll (shared/human.c CreateHuman): 14 capsule bones, transcribed as f32 literals. Bone order
// = the C's BoneId enum (pelvis, spine_01..03, neck, head, thigh/calf L/R, upper/lower_arm L/R).
type BoneSpec = {
    parent: number;
    refP: V3;
    refQ: Q4;
    cap1: V3;
    cap2: V3;
    radius: number;
    negGroup: boolean; // filter.groupIndex = -groupIndex (else 0)
    joint?: "spherical" | "revolute"; // pelvis (index 0) has none
    lfaP?: V3;
    lfaQ?: Q4;
    lfbP?: V3;
    lfbQ?: Q4;
    swingDeg?: number; // spherical cone half-angle, degrees
    twistDeg?: [number, number]; // twist (spherical) / limit (revolute), degrees
    friction: number; // jointFriction (× frictionTorque = maxMotorTorque)
};

const HUMAN_BONES: BoneSpec[] = [
    // pelvis
    {
        parent: -1,
        refP: [0.0, 0.932087, -0.051708],
        refQ: [0.739169, 0.0, 0.0, 0.67352],
        cap1: [0.07, 0.0, -0.08],
        cap2: [-0.07, 0.0, -0.08],
        radius: 0.13,
        negGroup: false,
        friction: 1.0,
    },
    // spine_01
    {
        parent: 0,
        refP: [0.0, 1.113505, -0.03481],
        refQ: [0.739973, 0.0, 0.0, 0.672637],
        cap1: [0.06, -0.0, -0.052264],
        cap2: [-0.06, 0.0, -0.052264],
        radius: 0.12,
        negGroup: true,
        joint: "spherical",
        lfaP: [0.0, 0.0, -0.182204],
        lfaQ: [-0.999999, 0.0, -0.0, 0.001194],
        lfbP: [0.0, 0.0, -0.007736],
        lfbQ: [-1.0, 0.0, -0.0, 0.0],
        swingDeg: 25.0,
        twistDeg: [-15.0, 15.0],
        friction: 1.0,
    },
    // spine_02
    {
        parent: 1,
        refP: [0.0, 1.194336, -0.027087],
        refQ: [0.703611, 0.0, 0.0, 0.710586],
        cap1: [0.08, -0.015133, -0.091801],
        cap2: [-0.08, -0.015133, -0.091801],
        radius: 0.1,
        negGroup: false,
        joint: "spherical",
        lfaP: [0.0, -0.0, -0.088935],
        lfaQ: [-0.998619, -0.0, 0.0, -0.05254],
        lfbP: [-0.0, 0.0, -0.008199],
        lfbQ: [-1.0, 0.0, -0.0, 0.0],
        swingDeg: 25.0,
        twistDeg: [-15.0, 15.0],
        friction: 1.0,
    },
    // spine_03
    {
        parent: 2,
        refP: [-0.0, 1.31043, -0.028232],
        refQ: [0.669856, 0.000001, -0.000001, 0.742491],
        cap1: [0.11, -0.039753, -0.13],
        cap2: [-0.11, -0.039753, -0.13],
        radius: 0.145,
        negGroup: false,
        joint: "spherical",
        lfaP: [-0.0, 0.0, -0.124298],
        lfaQ: [-0.998921, 0.000001, -0.000001, -0.046434],
        lfbP: [0.0, 0.0, 0.0],
        lfbQ: [-1.0, 0.0, -0.000001, 0.0],
        swingDeg: 15.0,
        twistDeg: [-10.0, 10.0],
        friction: 1.0,
    },
    // neck
    {
        parent: 3,
        refP: [0.0, 1.575582, -0.055837],
        refQ: [0.879922, 0.0, 0.0, 0.475118],
        cap1: [-0.000001, -0.0, -0.02],
        cap2: [0.0, -0.005, -0.08],
        radius: 0.07,
        negGroup: false,
        joint: "spherical",
        lfaP: [0.000001, -0.000259, -0.266585],
        lfaQ: [-0.942192, -0.000001, 0.0, 0.335074],
        lfbP: [0.0, 0.0, 0.0],
        lfbQ: [-1.0, 0.0, -0.000001, 0.0],
        swingDeg: 45.0,
        twistDeg: [-15.0, 15.0],
        friction: 0.8,
    },
    // head
    {
        parent: 4,
        refP: [0.0, 1.653348, -0.003241],
        refQ: [0.750288, 0.0, 0.0, 0.661111],
        cap1: [-0.000001, 0.016892, -0.05869],
        cap2: [0.0, -0.003629, -0.115072],
        radius: 0.0975,
        negGroup: false,
        joint: "spherical",
        lfaP: [0.0, 0.001321, -0.093873],
        lfaQ: [-0.974301, -0.0, -0.0, -0.225251],
        lfbP: [0.0, 0.001268, -0.005104],
        lfbQ: [-1.0, 0.0, -0.0, 0.0],
        swingDeg: 15.0,
        twistDeg: [-15.0, 15.0],
        friction: 0.4,
    },
    // thigh_l
    {
        parent: 0,
        refP: [0.090416, 0.986104, -0.03509],
        refQ: [-0.703287, -0.070715, 0.053866, 0.705327],
        cap1: [0.023719, 0.006008, -0.039068],
        cap2: [-0.064492, -0.004664, -0.424718],
        radius: 0.09,
        negGroup: true,
        joint: "spherical",
        lfaP: [0.05, 0.011537, -0.055325],
        lfaQ: [-0.714896, -0.022305, -0.698361, -0.02679],
        lfbP: [0.0, 0.0, 0.0],
        lfbQ: [-0.002064, 0.758987, 0.017046, 0.65088],
        swingDeg: 10.0,
        twistDeg: [-60.0, 40.0],
        friction: 1.0,
    },
    // calf_l
    {
        parent: 6,
        refP: [0.101198, 0.527027, -0.037374],
        refQ: [-0.653328, -0.06686, 0.058582, 0.751838],
        cap1: [0.001778, 0.0, 0.009841],
        cap2: [-0.078577, 0.014707, -0.41816],
        radius: 0.075,
        negGroup: false,
        joint: "revolute",
        lfaP: [-0.069989, 0.000253, -0.453844],
        lfaQ: [-0.000677, 0.760087, 0.105674, 0.641171],
        lfbP: [0.0, 0.0, 0.0],
        lfbQ: [-0.044589, 0.76554, 0.053368, 0.639619],
        twistDeg: [-5.0, 45.0],
        friction: 1.0,
    },
    // thigh_r
    {
        parent: 0,
        refP: [-0.090416, 0.986104, -0.03509],
        refQ: [-0.703287, 0.070715, -0.053865, 0.705326],
        cap1: [-0.023719, 0.006008, -0.039068],
        cap2: [0.064492, -0.004664, -0.424718],
        radius: 0.09,
        negGroup: true,
        joint: "spherical",
        lfaP: [-0.05, 0.011537, -0.055326],
        lfaQ: [-0.039089, -0.714094, 0.043177, 0.697623],
        lfbP: [0.0, 0.0, 0.0],
        lfbQ: [0.758805, -0.019886, -0.651012, -0.001759],
        swingDeg: 10.0,
        twistDeg: [-30.0, 60.0],
        friction: 1.0,
    },
    // calf_r
    {
        parent: 8,
        refP: [-0.101198, 0.527027, -0.037373],
        refQ: [-0.653327, 0.06686, -0.058582, 0.751839],
        cap1: [-0.00182, 0.0, 0.010071],
        cap2: [0.077883, 0.014825, -0.418047],
        radius: 0.075,
        negGroup: false,
        joint: "revolute",
        lfaP: [0.069988, 0.000253, -0.453844],
        lfaQ: [0.760086, -0.000675, -0.641171, -0.105676],
        lfbP: [0.0, 0.0, 0.0],
        lfbQ: [0.76554, -0.044589, -0.639619, -0.053368],
        twistDeg: [-45.0, 5.0],
        friction: 1.0,
    },
    // upper_arm_l
    {
        parent: 3,
        refP: [0.20378, 1.484275, -0.115897],
        refQ: [0.143082, 0.69598, -0.69013, 0.13733],
        cap1: [0.0, 0.0, 0.0],
        cap2: [-0.091118, 0.037775, 0.229719],
        radius: 0.075,
        negGroup: false,
        joint: "spherical",
        lfaP: [0.20378, -0.069369, -0.181921],
        lfaQ: [-0.278486, 0.4456, -0.097014, 0.845266],
        lfbP: [0.0, 0.0, 0.0],
        lfbQ: [-0.201396, -0.001586, 0.90185, 0.382234],
        swingDeg: 60.0,
        twistDeg: [-5.0, 5.0],
        friction: 1.0,
    },
    // lower_arm_l
    {
        parent: 10,
        refP: [0.305614, 1.242908, -0.117599],
        refQ: [0.165048, 0.563437, -0.802002, 0.109959],
        cap1: [0.0, 0.0, 0.0],
        cap2: [-0.142406, 0.039392, 0.261092],
        radius: 0.05,
        negGroup: false,
        joint: "revolute",
        lfaP: [-0.095482, 0.039584, 0.240723],
        lfaQ: [0.512487, -0.180629, 0.839474, 0.003742],
        lfbP: [0.0, 0.0, 0.0],
        lfbQ: [0.503803, -0.029831, 0.858168, 0.094017],
        twistDeg: [-5.0, 60.0],
        friction: 1.0,
    },
    // upper_arm_r
    {
        parent: 3,
        refP: [-0.20378, 1.484276, -0.115899],
        refQ: [0.143083, -0.695978, 0.690132, 0.137329],
        cap1: [0.0, 0.0, 0.0],
        cap2: [0.091118, 0.037775, 0.229718],
        radius: 0.075,
        negGroup: false,
        joint: "spherical",
        lfaP: [-0.203779, -0.069371, -0.181922],
        lfaQ: [-0.253621, -0.414842, 0.106962, 0.867261],
        lfbP: [0.0, 0.0, 0.0],
        lfbQ: [-0.201397, 0.001587, -0.90185, 0.382233],
        swingDeg: 60.0,
        twistDeg: [-5.0, 5.0],
        friction: 1.0,
    },
    // lower_arm_r
    {
        parent: 12,
        refP: [-0.305614, 1.242907, -0.117599],
        refQ: [0.165048, -0.563437, 0.802002, 0.109959],
        cap1: [0.0, 0.0, 0.0],
        cap2: [0.142406, 0.039392, 0.261092],
        radius: 0.05,
        negGroup: false,
        joint: "revolute",
        lfaP: [0.095484, 0.039585, 0.240723],
        lfaQ: [-0.180627, 0.512487, -0.003744, -0.839474],
        lfbP: [0.0, 0.0, 0.0],
        lfbQ: [-0.029831, 0.503803, -0.094017, -0.858169],
        twistDeg: [-60.0, 5.0],
        friction: 1.0,
    },
];

// Build a ragdoll at `position` (b3OffsetPos per bone), then its 13 spherical/revolute joints and the
// thigh_l/thigh_r filter joint. Frames are frounded to f32; joint local-frame quats are pre-normalized
// exactly as CreateHuman does. colorize/userData are dropped (they don't affect the sim/hash).
function createHuman(
    world: World,
    position: Vec3,
    frictionTorque: number,
    hertz: number,
    dampingRatio: number,
    groupIndex: number,
): void {
    const bodies: Body[] = [];
    for (const b of HUMAN_BONES) {
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: offsetPos(position, { x: f(b.refP[0]), y: f(b.refP[1]), z: f(b.refP[2]) }),
            rotation: { v: { x: f(b.refQ[0]), y: f(b.refQ[1]), z: f(b.refQ[2]) }, s: f(b.refQ[3]) },
        });
        body.createCapsule(
            {
                baseMaterial: { ...defaultSurfaceMaterial(), rollingResistance: f(0.2) },
                filter: { ...defaultFilter(), groupIndex: b.negGroup ? -groupIndex : 0 },
            },
            {
                center1: { x: f(b.cap1[0]), y: f(b.cap1[1]), z: f(b.cap1[2]) },
                center2: { x: f(b.cap2[0]), y: f(b.cap2[1]), z: f(b.cap2[2]) },
                radius: f(b.radius),
            },
        );
        bodies.push(body);
    }

    for (let i = 1; i < HUMAN_BONES.length; ++i) {
        const b = HUMAN_BONES[i];
        const bodyA = bodies[b.parent];
        const bodyB = bodies[i];
        const localFrameA = {
            p: { x: f(b.lfaP![0]), y: f(b.lfaP![1]), z: f(b.lfaP![2]) },
            q: quat.normalize({
                v: { x: f(b.lfaQ![0]), y: f(b.lfaQ![1]), z: f(b.lfaQ![2]) },
                s: f(b.lfaQ![3]),
            }),
        };
        const localFrameB = {
            p: { x: f(b.lfbP![0]), y: f(b.lfbP![1]), z: f(b.lfbP![2]) },
            q: quat.normalize({
                v: { x: f(b.lfbQ![0]), y: f(b.lfbQ![1]), z: f(b.lfbQ![2]) },
                s: f(b.lfbQ![3]),
            }),
        };
        const maxMotorTorque = f(f(b.friction) * frictionTorque);
        const enableSpring = hertz > 0;
        const twist = b.twistDeg as [number, number];
        if (b.joint === "revolute") {
            world.createRevoluteJoint(bodyA, bodyB, {
                localFrameA,
                localFrameB,
                enableLimit: true,
                lowerAngle: f(twist[0] * DEG_TO_RAD),
                upperAngle: f(twist[1] * DEG_TO_RAD),
                enableSpring,
                hertz,
                dampingRatio,
                enableMotor: true,
                maxMotorTorque,
            });
        } else {
            world.createSphericalJoint(bodyA, bodyB, {
                localFrameA,
                localFrameB,
                enableConeLimit: true,
                coneAngle: f((b.swingDeg as number) * DEG_TO_RAD),
                enableTwistLimit: true,
                lowerTwistAngle: f(twist[0] * DEG_TO_RAD),
                upperTwistAngle: f(twist[1] * DEG_TO_RAD),
                enableSpring,
                hertz,
                dampingRatio,
                enableMotor: true,
                maxMotorTorque,
            });
        }
    }

    // Disable thigh_l (6) / thigh_r (8) collision.
    world.createFilterJoint(bodies[6], bodies[8]);
}

const builders: Record<string, (world: World, fx: Fixture) => void> = {
    "free-fall": (world) => {
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 10, z: 0 },
            angularVelocity: { x: 2, y: 5, z: 1 },
        });
        body.createHull({}, makeBoxHull(0.5, 1.0, 1.5));
    },
    "sphere-drop": (world) => {
        createGround(world, 20.0);
        for (let i = 0; i < 5; ++i) {
            const body = world.createBody({
                type: BodyType.Dynamic,
                position: { x: 0, y: fround(1.0 + fround(1.5 * i)), z: 0 },
            });
            body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        }
    },
    "box-stack": (world) => {
        createGround(world, 20.0);
        for (let i = 0; i < 5; ++i) {
            const body = world.createBody({
                type: BodyType.Dynamic,
                position: { x: 0, y: fround(0.5 + fround(1.0 * i)), z: 0 },
            });
            body.createHull({}, makeBoxHull(0.5, 0.5, 0.5));
        }
    },
    "wake-drop": (world) => {
        createGround(world, 20.0);
        for (let i = 0; i < 2; ++i) {
            const body = world.createBody({
                type: BodyType.Dynamic,
                position: { x: 0, y: fround(0.5 + fround(1.0 * i)), z: 0 },
            });
            body.createHull({}, makeBoxHull(0.5, 0.5, 0.5));
        }
        const drop = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 8, z: 0 } });
        drop.createHull({}, makeBoxHull(0.5, 0.5, 0.5));
    },
    "split-slide": (world) => {
        createGround(world, 20.0);
        const left = world.createBody({
            type: BodyType.Dynamic,
            position: { x: -0.5, y: 0.5, z: 0 },
            linearVelocity: { x: -1, y: 0, z: 0 },
        });
        left.createHull({}, makeBoxHull(0.5, 0.5, 0.5));
        const right = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0.5, y: 0.5, z: 0 },
            linearVelocity: { x: 1, y: 0, z: 0 },
        });
        right.createHull({}, makeBoxHull(0.5, 0.5, 0.5));
    },
    "revolute-dd": (world) => {
        const b1 = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 5, z: 0 },
            angularDamping: 0.5,
        });
        b1.createHull({}, makeBoxHull(0.5, 0.25, 0.25));
        const b2 = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 1, y: 5, z: 0 },
            angularDamping: 0.5,
        });
        b2.createHull({}, makeBoxHull(0.5, 0.25, 0.25));
        world.createRevoluteJoint(b1, b2, {
            localFrameA: frame(0.5, 0, 0),
            localFrameB: frame(-0.5, 0, 0),
        });
    },
    "revolute-pendulum": (world) => {
        const anchor = world.createBody({ position: { x: 0, y: 5, z: 0 } });
        const arm = world.createBody({ type: BodyType.Dynamic, position: { x: 1, y: 5, z: 0 } });
        arm.createHull({}, makeBoxHull(1.0, 0.2, 0.2));
        world.createRevoluteJoint(anchor, arm, {
            localFrameA: frame(0, 0, 0),
            localFrameB: frame(-1, 0, 0),
        });
    },
    "revolute-motor": (world) => {
        const anchor = world.createBody({ position: { x: 0, y: 5, z: 0 } });
        const arm = world.createBody({ type: BodyType.Dynamic, position: { x: 1, y: 5, z: 0 } });
        arm.createHull({}, makeBoxHull(1.0, 0.2, 0.2));
        world.createRevoluteJoint(anchor, arm, {
            localFrameA: frame(0, 0, 0),
            localFrameB: frame(-1, 0, 0),
            enableMotor: true,
            motorSpeed: 3.0,
            maxMotorTorque: 1000.0,
        });
    },
    "revolute-limit": (world) => {
        const anchor = world.createBody({ position: { x: 0, y: 5, z: 0 } });
        const arm = world.createBody({ type: BodyType.Dynamic, position: { x: 1, y: 5, z: 0 } });
        arm.createHull({}, makeBoxHull(1.0, 0.2, 0.2));
        world.createRevoluteJoint(anchor, arm, {
            localFrameA: frame(0, 0, 0),
            localFrameB: frame(-1, 0, 0),
            enableLimit: true,
            lowerAngle: fround(-0.25 * PI),
            upperAngle: fround(0.25 * PI),
        });
    },
    "revolute-chain": (world) => {
        const anchor = world.createBody({ position: { x: 0, y: 5, z: 0 } });
        const arm1 = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0.5, y: 5, z: 0 },
            angularDamping: 0.5,
        });
        arm1.createHull({}, makeBoxHull(0.5, 0.15, 0.15));
        const arm2 = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 1.5, y: 5, z: 0 },
            angularDamping: 0.5,
        });
        arm2.createHull({}, makeBoxHull(0.5, 0.15, 0.15));
        world.createRevoluteJoint(anchor, arm1, {
            localFrameA: frame(0, 0, 0),
            localFrameB: frame(-0.5, 0, 0),
        });
        world.createRevoluteJoint(arm1, arm2, {
            localFrameA: frame(0.5, 0, 0),
            localFrameB: frame(-0.5, 0, 0),
        });
    },
    "weld-dd": (world) => {
        const b1 = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 5, z: 0 },
            angularDamping: 0.3,
        });
        b1.createHull({}, makeBoxHull(0.25, 0.25, 0.25));
        const b2 = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 1, y: 5, z: 0 },
            angularDamping: 0.3,
            angularVelocity: { x: 2, y: 0, z: 0 },
        });
        b2.createHull({}, makeBoxHull(0.25, 0.25, 0.25));
        world.createWeldJoint(b1, b2, {
            localFrameA: frame(0.5, 0, 0),
            localFrameB: frame(-0.5, 0, 0),
        });
    },
    parallel: (world) => {
        const b1 = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 5, z: 0 },
            angularDamping: 0.3,
        });
        b1.createHull({}, makeBoxHull(0.2, 0.2, 0.2));
        const b2 = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 1, y: 5, z: 0 },
            angularDamping: 0.3,
            angularVelocity: { x: 3, y: 2, z: 0 },
        });
        b2.createHull({}, makeBoxHull(0.2, 0.2, 0.2));
        world.createParallelJoint(b1, b2, { maxTorque: 5 });
    },
    // Eight free active-collinearity parallel-jointed pairs (one color, multiple GraphJoint blocks) plus a
    // row of overlapping dynamic boxes for contacts in the same colors — mirrors fixtures/gen.c
    // SceneJointContacts.
    "joint-contacts": (world) => {
        for (let k = 0; k < 8; ++k) {
            const x = k * 3;
            const a = world.createBody({
                type: BodyType.Dynamic,
                position: { x, y: 8, z: 0 },
                angularDamping: 0.3,
            });
            a.createHull({}, makeBoxHull(0.2, 0.2, 0.2));
            const b = world.createBody({
                type: BodyType.Dynamic,
                position: { x: x + 1, y: 8, z: 0 },
                angularDamping: 0.3,
                angularVelocity: { x: 3, y: 2, z: 0 },
            });
            b.createHull({}, makeBoxHull(0.2, 0.2, 0.2));
            world.createParallelJoint(a, b, { maxTorque: 5 });
        }
        for (let i = 0; i < 12; ++i) {
            const d = world.createBody({
                type: BodyType.Dynamic,
                position: { x: i * 0.484375, y: 5, z: 0 },
                angularVelocity: { x: 0, y: 0, z: i % 2 === 0 ? 2 : -2 },
            });
            d.createHull({}, makeBoxHull(0.25, 0.25, 0.25));
        }
    },
    motor: (world) => {
        const anchor = world.createBody({ position: { x: 0, y: 5, z: 0 } });
        const arm = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 1, y: 5, z: 0 },
            angularDamping: 0.2,
        });
        arm.createHull({}, makeBoxHull(1.0, 0.2, 0.2));
        world.createMotorJoint(anchor, arm, {
            localFrameA: frame(0, 0, 0),
            localFrameB: frame(-1, 0, 0),
            maxVelocityForce: 1000,
            angularVelocity: { x: 0, y: 0, z: 2 },
            maxVelocityTorque: 200,
        });
    },
    "motor-spring": (world) => {
        const anchor = world.createBody({ position: { x: 0, y: 5, z: 0 } });
        const arm = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 1, y: 5, z: 0 },
            angularDamping: 0.2,
        });
        arm.createHull({}, makeBoxHull(1.0, 0.2, 0.2));
        world.createMotorJoint(anchor, arm, {
            localFrameA: frame(0, 0, 0),
            localFrameB: frame(-1, 0, 0),
            maxSpringForce: 1000,
            linearHertz: 5,
            linearDampingRatio: 0.7,
            maxSpringTorque: 200,
            angularHertz: 5,
            angularDampingRatio: 0.7,
        });
    },
    distance: (world) => {
        const anchor = world.createBody({ position: { x: 0, y: 5, z: 0 } });
        const ball = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 2, y: 5, z: 0 },
            linearDamping: 0.1,
        });
        ball.createHull({}, makeBoxHull(0.2, 0.2, 0.2));
        world.createDistanceJoint(anchor, ball, { length: 2 });
    },
    "distance-spring": (world) => {
        const anchor = world.createBody({ position: { x: 0, y: 5, z: 0 } });
        const ball = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 3, z: 0 } });
        ball.createHull({}, makeBoxHull(0.2, 0.2, 0.2));
        world.createDistanceJoint(anchor, ball, {
            length: 2,
            enableSpring: true,
            hertz: 3,
            dampingRatio: 0.3,
            enableLimit: true,
            minLength: 1,
            maxLength: 3,
            enableMotor: true,
            motorSpeed: 0.5,
            maxMotorForce: 10,
        });
    },
    prismatic: (world) => {
        const anchor = world.createBody({ position: { x: 0, y: 5, z: 0 } });
        const slider = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 5, z: 0 },
            linearVelocity: { x: 3, y: 0, z: 0 },
        });
        slider.createHull({}, makeBoxHull(0.25, 0.25, 0.25));
        world.createPrismaticJoint(anchor, slider, {
            enableSpring: true,
            hertz: 3,
            dampingRatio: 0.3,
            enableLimit: true,
            lowerTranslation: -1,
            upperTranslation: 1,
        });
    },
    "prismatic-motor": (world) => {
        const anchor = world.createBody({ position: { x: 0, y: 5, z: 0 } });
        const slider = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 5, z: 0 } });
        slider.createHull({}, makeBoxHull(0.25, 0.25, 0.25));
        world.createPrismaticJoint(anchor, slider, {
            enableMotor: true,
            motorSpeed: 2,
            maxMotorForce: 50,
            enableLimit: true,
            lowerTranslation: -2,
            upperTranslation: 2,
        });
    },
    spherical: (world) => {
        const anchor = world.createBody({ position: { x: 0, y: 5, z: 0 } });
        const arm = world.createBody({ type: BodyType.Dynamic, position: { x: 1, y: 5, z: 0 } });
        arm.createHull({}, makeBoxHull(1.0, 0.2, 0.2));
        world.createSphericalJoint(anchor, arm, { localFrameB: frame(-1, 0, 0) });
    },
    "spherical-limits": (world) => {
        const anchor = world.createBody({ position: { x: 0, y: 5, z: 0 } });
        const arm = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 1, y: 5, z: 0 },
            angularVelocity: { x: 3, y: 0, z: 4 },
            angularDamping: 0.2,
        });
        arm.createHull({}, makeBoxHull(1.0, 0.2, 0.2));
        world.createSphericalJoint(anchor, arm, {
            localFrameB: frame(-1, 0, 0),
            enableConeLimit: true,
            coneAngle: fround(fround(0.4) * PI),
            enableTwistLimit: true,
            lowerTwistAngle: fround(fround(-0.3) * PI),
            upperTwistAngle: fround(fround(0.3) * PI),
        });
    },
    "spherical-motor": (world) => {
        const anchor = world.createBody({ position: { x: 0, y: 5, z: 0 } });
        const arm = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 1, y: 5, z: 0 },
            angularDamping: 0.2,
        });
        arm.createHull({}, makeBoxHull(1.0, 0.2, 0.2));
        world.createSphericalJoint(anchor, arm, {
            localFrameB: frame(-1, 0, 0),
            enableSpring: true,
            hertz: 5,
            dampingRatio: 0.7,
            enableMotor: true,
            motorVelocity: { x: 0, y: 0, z: 3 },
            maxMotorTorque: 100,
        });
    },
    wheel: (world) => {
        const chassis = world.createBody({ position: { x: 0, y: 5, z: 0 } });
        const wheel = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 5, z: 0 },
            linearVelocity: { x: 2, y: 0, z: 0 },
        });
        wheel.createHull({}, makeBoxHull(0.3, 0.3, 0.3));
        world.createWheelJoint(chassis, wheel, {});
    },
    "wheel-spin": (world) => {
        const chassis = world.createBody({ position: { x: 0, y: 5, z: 0 } });
        const wheel = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 5, z: 0 } });
        wheel.createHull({}, makeBoxHull(0.3, 0.3, 0.3));
        world.createWheelJoint(chassis, wheel, {
            enableSpinMotor: true,
            spinSpeed: 10,
            maxSpinTorque: 50,
        });
    },
    "wheel-steer": (world) => {
        const chassis = world.createBody({ position: { x: 0, y: 5, z: 0 } });
        const wheel = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 5, z: 0 },
            linearVelocity: { x: 1, y: 0, z: 0 },
        });
        wheel.createHull({}, makeBoxHull(0.3, 0.3, 0.3));
        world.createWheelJoint(chassis, wheel, {
            enableSteering: true,
            targetSteeringAngle: 0.3,
            maxSteeringTorque: 50,
            enableSteeringLimit: true,
            lowerSteeringLimit: -0.5,
            upperSteeringLimit: 0.5,
            enableSuspensionLimit: true,
            lowerSuspensionLimit: -1,
            upperSuspensionLimit: 1,
        });
    },
    ragdoll: (world) => {
        createGround(world, 20.0);
        const torso = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 3, z: 0 },
            angularDamping: 0.5,
            linearDamping: 0.5,
        });
        torso.createHull({}, makeBoxHull(0.25, 0.5, 0.25));

        const armL = world.createBody({
            type: BodyType.Dynamic,
            position: { x: -0.75, y: 3.5, z: 0 },
            angularDamping: 0.5,
            linearDamping: 0.5,
        });
        armL.createHull({}, makeBoxHull(0.5, 0.125, 0.125));
        const armR = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0.75, y: 3.5, z: 0 },
            angularDamping: 0.5,
            linearDamping: 0.5,
        });
        armR.createHull({}, makeBoxHull(0.5, 0.125, 0.125));
        world.createSphericalJoint(torso, armL, {
            localFrameA: frame(-0.25, 0.5, 0),
            localFrameB: frame(0.5, 0, 0),
        });
        world.createSphericalJoint(torso, armR, {
            localFrameA: frame(0.25, 0.5, 0),
            localFrameB: frame(-0.5, 0, 0),
        });

        const legL = world.createBody({
            type: BodyType.Dynamic,
            position: { x: -0.25, y: 2.0, z: 0 },
            angularDamping: 0.5,
            linearDamping: 0.5,
        });
        legL.createHull({}, makeBoxHull(0.125, 0.5, 0.125));
        const legR = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0.25, y: 2.0, z: 0 },
            angularDamping: 0.5,
            linearDamping: 0.5,
        });
        legR.createHull({}, makeBoxHull(0.125, 0.5, 0.125));
        world.createRevoluteJoint(torso, legL, {
            localFrameA: frame(-0.25, -0.5, 0),
            localFrameB: frame(0, 0.5, 0),
        });
        world.createRevoluteJoint(torso, legR, {
            localFrameA: frame(0.25, -0.5, 0),
            localFrameB: frame(0, 0.5, 0),
        });
    },
    "ccd-drop": (world) => {
        // Thin static floor (top at y = 0); -0.05 isn't f32-exact, so fround it to match C's -0.05f.
        const ground = world.createBody({ position: { x: 0, y: fround(-0.05), z: 0 } });
        ground.createHull({}, makeBoxHull(5.0, 0.05, 5.0));
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 5, z: 0 },
            linearVelocity: { x: 0, y: -40, z: 0 },
            angularVelocity: { x: 1, y: 0, z: 0.5 },
        });
        body.createHull({}, makeBoxHull(0.25, 0.25, 0.25));
    },
    "ccd-bullet": (world) => {
        createGround(world, 20.0);
        const wall = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 1, z: 0 } });
        wall.createHull({}, makeBoxHull(0.05, 1.0, 1.0));
        const bullet = world.createBody({
            type: BodyType.Dynamic,
            isBullet: true,
            position: { x: -5, y: 1, z: 0 },
            linearVelocity: { x: 200, y: 0, z: 0 },
        });
        bullet.createHull({}, makeBoxHull(0.1, 0.1, 0.1));
    },
    "mesh-box": (world) => {
        createMeshFloor(world);
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 2, z: 0 },
            angularVelocity: { x: 0.25, y: 0, z: 0.5 },
        });
        body.createHull({}, makeBoxHull(1.0, 1.0, 1.0));
    },
    "mesh-sphere": (world) => {
        createMeshFloor(world);
        const body = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 2, z: 0 } });
        body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
    },
    "mesh-capsule": (world) => {
        createMeshFloor(world);
        const body = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 2, z: 0 } });
        body.createCapsule(
            {},
            { center1: { x: -0.5, y: 0, z: 0 }, center2: { x: 0.5, y: 0, z: 0 }, radius: 0.25 },
        );
    },
    "mesh-ccd": (world) => {
        createMeshFloor(world);
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 5, z: 0 },
            linearVelocity: { x: 0, y: -40, z: 0 },
            angularVelocity: { x: 1, y: 0, z: 0.5 },
        });
        body.createHull({}, makeBoxHull(0.25, 0.25, 0.25));
    },
    "height-box": (world) => {
        createHeightFieldFloor(world);
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 2, z: 0 },
            angularVelocity: { x: 0.25, y: 0, z: 0.5 },
        });
        body.createHull({}, makeBoxHull(1.0, 1.0, 1.0));
    },
    "height-sphere": (world) => {
        createHeightFieldFloor(world);
        const body = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 2, z: 0 } });
        body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
    },
    "height-capsule": (world) => {
        createHeightFieldFloor(world);
        const body = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 2, z: 0 } });
        body.createCapsule(
            {},
            { center1: { x: -0.5, y: 0, z: 0 }, center2: { x: 0.5, y: 0, z: 0 }, radius: 0.25 },
        );
    },
    "height-ccd": (world) => {
        createHeightFieldFloor(world);
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 5, z: 0 },
            linearVelocity: { x: 0, y: -40, z: 0 },
            angularVelocity: { x: 1, y: 0, z: 0.5 },
        });
        body.createHull({}, makeBoxHull(0.25, 0.25, 0.25));
    },
    "compound-hull": (world) => {
        const mat = defaultSurfaceMaterial();
        const slab = makeBoxHull(1.5, 0.25, 1.5);
        const compound = createCompound({
            hulls: [
                {
                    hull: slab,
                    transform: { p: { x: 1.4, y: -0.25, z: 0 }, q: QUAT_ID },
                    material: mat,
                },
                {
                    hull: slab,
                    transform: { p: { x: -1.4, y: -0.25, z: 0 }, q: QUAT_ID },
                    material: mat,
                },
            ],
        });
        const floor = world.createBody({});
        floor.createCompound({}, compound as CompoundData);
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 2, z: 0 },
            angularVelocity: { x: 0.25, y: 0, z: 0.5 },
        });
        body.createHull({}, makeBoxHull(1.0, 1.0, 1.0));
    },
    "compound-capsule": (world) => {
        const mat = defaultSurfaceMaterial();
        const compound = createCompound({
            capsules: [
                {
                    capsule: {
                        center1: { x: -2, y: 0, z: 0.7 },
                        center2: { x: 2, y: 0, z: 0.7 },
                        radius: 0.3,
                    },
                    material: mat,
                },
                {
                    capsule: {
                        center1: { x: -2, y: 0, z: -0.7 },
                        center2: { x: 2, y: 0, z: -0.7 },
                        radius: 0.3,
                    },
                    material: mat,
                },
            ],
        });
        const floor = world.createBody({});
        floor.createCompound({}, compound as CompoundData);
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 2, z: 0 },
            angularVelocity: { x: 0, y: 0, z: 0.3 },
        });
        body.createHull({}, makeBoxHull(1.0, 1.0, 1.0));
    },
    "compound-sphere": (world) => {
        const mat = defaultSurfaceMaterial();
        const spheres = [];
        for (let ix = -1; ix <= 1; ++ix) {
            for (let iz = -1; iz <= 1; ++iz) {
                spheres.push({
                    sphere: { center: { x: ix, y: 0, z: iz }, radius: 0.5 },
                    material: mat,
                });
            }
        }
        const compound = createCompound({ spheres });
        const floor = world.createBody({});
        floor.createCompound({}, compound as CompoundData);
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 2, z: 0 },
            angularVelocity: { x: 0.25, y: 0, z: 0.5 },
        });
        body.createHull({}, makeBoxHull(1.0, 1.0, 1.0));
    },
    "compound-mesh": (world) => {
        const mat = defaultSurfaceMaterial();
        const mesh = createGridMesh(8, 8, 1.0, 0, true);
        const compound = createCompound({
            meshes: [
                {
                    meshData: mesh,
                    transform: { p: { x: 0, y: -0.5, z: 0 }, q: QUAT_ID },
                    scale: { x: 1, y: 1, z: 1 },
                    materials: [mat],
                    materialCount: 1,
                },
            ],
        });
        const floor = world.createBody({});
        floor.createCompound({}, compound as CompoundData);
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 2, z: 0 },
            angularVelocity: { x: 0.25, y: 0, z: 0.5 },
        });
        body.createHull({}, makeBoxHull(1.0, 1.0, 1.0));
    },
    "compound-ccd": (world) => {
        const mat = defaultSurfaceMaterial();
        const slab = makeBoxHull(1.5, 0.25, 1.5);
        const compound = createCompound({
            hulls: [
                {
                    hull: slab,
                    transform: { p: { x: 1.4, y: -0.25, z: 0 }, q: QUAT_ID },
                    material: mat,
                },
                {
                    hull: slab,
                    transform: { p: { x: -1.4, y: -0.25, z: 0 }, q: QUAT_ID },
                    material: mat,
                },
            ],
        });
        const floor = world.createBody({});
        floor.createCompound({}, compound as CompoundData);
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 5, z: 0 },
            linearVelocity: { x: 0, y: -40, z: 0 },
            angularVelocity: { x: 1, y: 0, z: 0.5 },
        });
        body.createHull({}, makeBoxHull(0.25, 0.25, 0.25));
    },
    sensor: (world) => {
        // A static box-hull sensor volume (y in [1, 5]) with dynamic spheres falling through it and a
        // fast box sweeping through it. The sensor is non-solid, so the body hashes must match a
        // no-sensor run bit-for-bit — the overlap pass + continuous sensor branch leave dynamics alone.
        createGround(world, 20.0);

        const sensorBody = world.createBody({ position: { x: 0, y: 3, z: 0 } });
        sensorBody.createHull(
            { isSensor: true, enableSensorEvents: true },
            makeBoxHull(4.0, 2.0, 4.0),
        );

        const xs = [-1.5, 0.0, 1.5];
        const ys = [6.0, 8.0, 10.0];
        for (let i = 0; i < 3; ++i) {
            const body = world.createBody({
                type: BodyType.Dynamic,
                position: { x: xs[i], y: ys[i], z: 0 },
            });
            body.createSphere(
                { enableSensorEvents: true },
                { center: { x: 0, y: 0, z: 0 }, radius: 0.5 },
            );
        }

        const fast = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 3, y: 13, z: 0 },
            linearVelocity: { x: 0, y: -55, z: 0 },
        });
        fast.createHull({ enableSensorEvents: true }, makeBoxHull(0.25, 0.25, 0.25));
    },
    // Stage 14 benchmark scenes at reduced scale (mirror fixtures/gen.c's SceneBench* exactly). They
    // extend the bit-exact contract to scale; f32 discipline follows the port rule (fround per op, one
    // op per wrap). Values fed through the ported math helpers (quat/vec3) are already f32-rounded.
    "bench-pyramid": (world) => {
        const baseCount = 16;
        const ground = world.createBody({ position: { x: 0, y: -1, z: 0 } });
        ground.createHull({}, makeBoxHull(100.0, 1.0, 100.0));

        const h = 0.5;
        const box = makeBoxHull(h, h, h);
        const shift = fround(1.0 * h);
        for (let i = 0; i < baseCount; ++i) {
            const y = fround(fround(fround(2.0 * i) + 1.0) * shift);
            for (let j = i; j < baseCount; ++j) {
                const x = fround(
                    fround(
                        fround(fround(i + 1.0) * shift) + fround(fround(2.0 * (j - i)) * shift),
                    ) - fround(h * baseCount),
                );
                const body = world.createBody({ type: BodyType.Dynamic, position: { x, y, z: 0 } });
                body.createHull({ density: 100.0 }, box);
            }
        }
    },
    "bench-many-pyramids": (world) => {
        const baseCount = 4;
        const extent = 0.5;
        const rowCount = 3;
        const columnCount = 3;
        const groundExtent = fround(fround(extent * columnCount) * fround(baseCount + 1.0));
        const ground = world.createBody({ position: { x: 0, y: -1, z: 0 } });
        ground.createHull({}, makeBoxHull(groundExtent, 1.0, groundExtent));

        const box = makeBoxHull(extent, extent, extent);
        const smallPyramid = (centerX: number, baseZ: number) => {
            for (let i = 0; i < baseCount; ++i) {
                const y = fround(fround(fround(2.0 * i) + 1.0) * extent);
                for (let j = i; j < baseCount; ++j) {
                    // C: (i+1)*extent + 2*(j-i)*extent + centerX - 0.5, left-assoc.
                    const x = fround(
                        fround(
                            fround(
                                fround(fround(i + 1.0) * extent) +
                                    fround(fround(2.0 * (j - i)) * extent),
                            ) + centerX,
                        ) - 0.5,
                    );
                    const body = world.createBody({
                        type: BodyType.Dynamic,
                        position: { x, y, z: baseZ },
                    });
                    body.createHull({ density: 100.0 }, box);
                }
            }
        };

        const baseWidth = fround(fround(2.0 * extent) * baseCount);
        let baseZ = fround(-groundExtent + fround(2.0 * extent));
        const deltaZ = fround(
            fround(2.0 * fround(groundExtent - fround(2.0 * extent))) / fround(rowCount - 1.0),
        );
        for (let i = 0; i < rowCount; ++i) {
            for (let j = 0; j < columnCount; ++j) {
                const centerX = fround(
                    fround(-groundExtent + fround(j * fround(baseWidth + fround(2.0 * extent)))) +
                        fround(2.0 * extent),
                );
                smallPyramid(centerX, baseZ);
            }
            baseZ = fround(baseZ + deltaZ);
        }
    },
    "bench-joint-grid": (world) => {
        const n = 10;
        const bodies: Body[] = [];
        const filter = { categoryBits: 2n, maskBits: 0xffffffffn ^ 2n, groupIndex: 0 };
        const sphere = { center: { x: 0, y: 0, z: 0 }, radius: 0.4 };
        for (let k = 0; k < n; ++k) {
            for (let i = 0; i < n; ++i) {
                const body = world.createBody({
                    type: i === 0 ? BodyType.Static : BodyType.Dynamic,
                    position: { x: k, y: -i, z: 0 },
                });
                body.createSphere({ filter }, sphere);
                const index = bodies.length;
                if (i > 0) {
                    world.createSphericalJoint(bodies[index - 1], body, {
                        localFrameA: frame(0, -0.5, 0),
                        localFrameB: frame(0, 0.5, 0),
                    });
                }
                if (k > 0) {
                    world.createSphericalJoint(bodies[index - n], body, {
                        localFrameA: frame(0.5, 0, 0),
                        localFrameB: frame(-0.5, 0, 0),
                    });
                }
                bodies.push(body);
            }
        }
    },
    "bench-washer": (world) => {
        const ground = world.createBody({ position: { x: 0, y: -1, z: 0 } });
        ground.createHull({}, makeBoxHull(60.0, 1.0, 60.0));

        const motorSpeed = 25.0;
        const washer = world.createBody({
            type: BodyType.Kinematic,
            position: { x: 0, y: 21, z: 0 },
            angularVelocity: { x: 0, y: 0, z: fround(fround(PI / 180.0) * motorSpeed) },
            linearVelocity: { x: 0.001, y: -0.002, z: 0 },
        });

        const r0 = 14.0;
        const r1 = 16.0;
        const r2 = 18.0;
        const nd = { x: 0, y: 0, z: -10.0 };
        const pd = { x: 0, y: 0, z: 10.0 };
        const axisZ = { x: 0, y: 0, z: 1 };
        const angle = fround(PI / 18.0);
        const q = quat.fromAxisAngle(axisZ, angle);
        const qo = quat.fromAxisAngle(axisZ, fround(fround(0.1) * angle));
        let u1 = { x: 1, y: 0, z: 0 };
        for (let i = 0; i < 36; ++i) {
            const u2 = i === 35 ? { x: 1, y: 0, z: 0 } : quat.rotate(q, u1);
            {
                const a1 = quat.invRotate(qo, u1);
                const a2 = quat.rotate(qo, u2);
                const points = [
                    vec3.mulAdd(nd, r1, a1),
                    vec3.mulAdd(nd, r2, a1),
                    vec3.mulAdd(nd, r1, a2),
                    vec3.mulAdd(nd, r2, a2),
                    vec3.mulAdd(pd, r1, a1),
                    vec3.mulAdd(pd, r2, a1),
                    vec3.mulAdd(pd, r1, a2),
                    vec3.mulAdd(pd, r2, a2),
                ];
                const hull = createHull(points, 8);
                if (hull) washer.createHull({}, hull);
            }
            if (i % 9 === 0) {
                const points = [
                    vec3.mulAdd(nd, r0, u1),
                    vec3.mulAdd(nd, r1, u1),
                    vec3.mulAdd(nd, r0, u2),
                    vec3.mulAdd(nd, r1, u2),
                    vec3.mulAdd(pd, r0, u1),
                    vec3.mulAdd(pd, r1, u1),
                    vec3.mulAdd(pd, r0, u2),
                    vec3.mulAdd(pd, r1, u2),
                ];
                const hull = createHull(points, 8);
                if (hull) washer.createHull({}, hull);
            }
            u1 = u2;
        }

        const a = fround(0.2);
        const gridCount = 4;
        const cube = makeBoxHull(a, a, a);
        const step = fround(4.0 * a);
        let x = fround(fround(-2.0 * a) * gridCount);
        for (let i = 0; i < gridCount; ++i) {
            let y = fround(fround(fround(-2.0 * a) * gridCount) + 21.0);
            for (let j = 0; j < gridCount; ++j) {
                let z = fround(fround(-2.0 * a) * gridCount);
                for (let k = 0; k < gridCount; ++k) {
                    const body = world.createBody({
                        type: BodyType.Dynamic,
                        position: { x, y, z },
                    });
                    body.createHull({}, cube);
                    z = fround(z + step);
                }
                y = fround(y + step);
            }
            x = fround(x + step);
        }
    },
    "bench-large-world": (world) => {
        const cell = BENCH_LW_CELL;
        const gridCount = BENCH_LW_GRID;
        const halfSpan = fround(fround(0.5 * cell) * gridCount);
        const box = makeBoxHull(fround(0.5 * cell), 0.25, fround(0.5 * cell));
        for (let i = 0; i < gridCount; ++i) {
            const x = fround(-halfSpan + fround(fround(i + 0.5) * cell));
            for (let j = 0; j < gridCount; ++j) {
                const z = fround(-halfSpan + fround(fround(j + 0.5) * cell));
                const body = world.createBody({ position: { x, y: 0, z } });
                body.createHull({ invokeContactCreation: true }, box);
            }
        }
    },
    "bench-trees": (world, fx) => {
        // Ground: rebuild the libm-sinf wave mesh from the C-emitted vertices (option A) so the
        // dynamics stay bit-exact by the double-rounding theorem. The triangle topology is pure integer
        // grid indexing (no sinf), so the port computes it. CreateTrees uses tilt = 0, so the ground
        // rotation is identity and every tree base sits at y = 1.0.
        const gv = fx.groundVertices as number[][];
        const vertices = gv.map(([x, y, z]) => ({ x, y, z }));
        const indices: number[] = [];
        for (let ix = 0; ix < TREES_XCOUNT; ++ix) {
            for (let iz = 0; iz < TREES_ZCOUNT; ++iz) {
                const i1 = iz + (TREES_ZCOUNT + 1) * ix;
                const i2 = i1 + 1;
                const i3 = i2 + (TREES_ZCOUNT + 1);
                const i4 = i3 - 1;
                indices.push(i1, i2, i3, i3, i4, i1);
            }
        }
        const mesh = createMesh({ vertices, indices, useMedianSplit: true, identifyEdges: true });
        if (!mesh) throw new Error("bench-trees: ground mesh build failed");
        const ground = world.createBody({ position: { x: 0, y: 0, z: 0 } });
        ground.createMesh({}, mesh, { x: 1, y: 1, z: 1 });

        // Trees: tapering stacks of 22 cylinders (portable-trig hulls the port builds itself).
        const hulls: HullData[] = [];
        let y = 1.0;
        let r = 0.75;
        const l = 1.5;
        for (let i = 0; i < 22; ++i) {
            const h = fround(l + fround(2.0 * r));
            hulls.push(createCylinder(h, r, fround(y - r), 6));
            y = fround(y + h);
            // fround the taper literal so the multiplier is C's f32 0.95f, not the f64 0.95 — the
            // 1-ULP difference compounds through the taper and diverges the hulls from cylinder 11 on.
            r = fround(fround(0.95) * r);
        }

        const treeDef = {
            baseMaterial: {
                ...defaultSurfaceMaterial(),
                friction: fround(0.9),
                rollingResistance: fround(0.05),
            },
            updateBodyMass: false,
            density: 1.0,
        };
        let angularVelocity = -0.5;
        let z = -15.0;
        for (let bodyIndex = 0; bodyIndex < TREES_BODIES; ++bodyIndex) {
            const pos = { x: 0, y: 1.0, z };
            const body = world.createBody({
                type: BodyType.Dynamic,
                position: pos,
                sleepThreshold: fround(0.2),
            });
            for (let s = 0; s < 22; ++s) body.createHull(treeDef, hulls[s]);

            const velocityScale = fround(0.5 + fround(fround(0.5 * bodyIndex) / TREES_BODIES));
            body.applyMassFromShapes();
            const center = body.getWorldCenterOfMass();
            const omega = { x: 0, y: 0, z: fround(velocityScale * angularVelocity) };
            const v = vec3.cross(omega, vec3.sub(center, pos));
            body.setAngularVelocity(omega);
            body.setLinearVelocity(v);

            z = fround(z + 3.0);
            angularVelocity = -angularVelocity;
        }
    },
    "bench-junkyard": (world) => {
        // Ground box (top at y = 0) with four walls, then a 3x3 rock pile near the origin. The
        // kinematic pusher is created last (matching gen.c's body order) and swept by the stepFactory.
        const ground = world.createBody({ position: { x: 0, y: -1, z: 0 } });
        ground.createHull({}, makeBoxHull(120.0, 1.0, 120.0));
        ground.createHull({}, makeOffsetBoxHull(1.0, 8.0, 50.0, { x: -50, y: 8, z: 0 }));
        ground.createHull({}, makeOffsetBoxHull(1.0, 8.0, 50.0, { x: 50, y: 8, z: 0 }));
        ground.createHull({}, makeOffsetBoxHull(50.0, 8.0, 1.0, { x: 0, y: 8, z: -50 }));
        ground.createHull({}, makeOffsetBoxHull(50.0, 8.0, 1.0, { x: 0, y: 8, z: 50 }));

        const rockHull = createRock(1.5);
        for (let X = 0; X < JUNK_ROCK_COUNT; ++X) {
            for (let Z = 0; Z < JUNK_ROCK_COUNT; ++Z) {
                const body = world.createBody({
                    type: BodyType.Dynamic,
                    position: {
                        x: fround(JUNK_ROCK_BASE + fround(JUNK_ROCK_SPACING * X)),
                        y: fround(JUNK_ROCK_HEIGHT + 1.0),
                        z: fround(JUNK_ROCK_BASE + fround(JUNK_ROCK_SPACING * Z)),
                    },
                });
                body.createHull({}, rockHull);
            }
        }

        junkyardPusher = world.createBody({
            type: BodyType.Kinematic,
            position: { x: JUNK_RADIUS, y: 0, z: 0 },
        });
        junkyardPusher.createHull({}, createCylinder(24.0, 4.0, 0.0, 16));
    },
    "bench-rain": (world) => {
        // Static mesh ground (grid + torus, portable-trig → bit-exact). Humans spawn over time via the
        // stepFactory. One ground tile at the origin (reduced from the benchmark's grid of tiles).
        const grid = createGridMesh(
            2 * RAINF_GROUND_HALF,
            2 * RAINF_GROUND_HALF,
            RAINF_CELL,
            1,
            true,
        );
        const torus = createTorusMesh(16, 16, RAINF_TORUS_MAJOR, 1.0);
        const ground = world.createBody({});
        ground.createMesh({}, grid, { x: 1, y: 1, z: 1 });
        ground.createMesh({}, torus, { x: 1, y: 1, z: 1 });
    },
};

// The kinematic pusher handle, set by the bench-junkyard builder and driven by its stepFactory
// (mirrors gen.c's file-static g_junkyardPusher). Tests run serially, so a module-level holder is safe.
let junkyardPusher: Body | null = null;

// The sleep scenes reuse the pile builders; only the world's enableSleep differs (as in gen.c).
const sceneBuilder: Record<string, string> = {
    "sphere-sleep": "sphere-drop",
    "box-sleep": "box-stack",
    drift: "box-stack",
};

// Scenes that spawn bodies over time drive a per-step hook, called with the loop index before each
// world.step (mirroring gen.c's stepFn(i) → Step). Each factory returns a fresh, stateful stepper per
// run. Mirrors fixtures/gen.c's StepBench* exactly.
const stepFactories: Record<string, () => (world: World, step: number) => void> = {
    "bench-large-world": () => {
        let dropped = 0;
        let side = 1;
        while (side * side < BENCH_LW_SPHERES) side += 1;
        const halfSpan = fround(fround(0.5 * BENCH_LW_CELL) * BENCH_LW_GRID);
        const inset = fround(fround(fround(0.1) * 2.0) * halfSpan);
        const usable = fround(fround(2.0 * halfSpan) - fround(2.0 * inset));
        const step = fround(usable / side);
        return (world, stepCount) => {
            if (dropped >= BENCH_LW_SPHERES) return;
            if (stepCount === 0) return;
            if (stepCount % BENCH_LW_INTERVAL !== 0) return;

            const gi = dropped % side;
            const gj = Math.floor(dropped / side);
            const x = fround(fround(-halfSpan + inset) + fround(fround(gi + 0.5) * step));
            const z = fround(fround(-halfSpan + inset) + fround(fround(gj + 0.5) * step));
            const body = world.createBody({
                type: BodyType.Dynamic,
                position: { x, y: 1.5, z },
            });
            body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
            dropped += 1;
        };
    },
    "bench-junkyard": () => {
        // Sweep the kinematic pusher along a circle via setTargetTransform (mirrors StepBenchJunkyard).
        // The target rotation is identity, so only the linear-velocity path runs. timeStep is the
        // scene's own 1/60, matching the world step dt.
        let degrees = 0;
        const timeStep = fround(1 / 60);
        const omega = -6.0;
        return () => {
            degrees = fround(degrees + fround(omega * timeStep));
            const cs = computeCosSin(fround(fround(degrees * PI) / 180.0));
            const target = {
                p: { x: fround(JUNK_RADIUS * cs.cosine), y: 0, z: fround(JUNK_RADIUS * cs.sine) },
                q: QUAT_ID,
            };
            junkyardPusher?.setTargetTransform(target, timeStep, false);
        };
    },
    "bench-rain": () => {
        // Spawn a ragdoll every RAINF_SPAWN_INTERVAL steps at descending x with a distinct group index
        // (mirrors StepBenchRain). Creating bodies AND joints mid-replay is rain's unique coverage.
        let spawned = 0;
        return (world, stepCount) => {
            if (spawned < RAINF_HUMAN_COUNT && stepCount % RAINF_SPAWN_INTERVAL === 0) {
                const position = { x: f(-3.0 + f(3.0 * spawned)), y: 8.0, z: 0.0 };
                createHuman(world, position, 5.0, 1.0, 0.7, spawned + 1);
                spawned += 1;
            }
        };
    },
};

// scene name → [enableSleep, enableContinuous], matching gen.c's per-scene world flags.
const SCENES: [string, boolean, boolean][] = [
    ["free-fall", false, false],
    ["sphere-drop", false, false],
    ["box-stack", false, false],
    ["sphere-sleep", true, false],
    ["box-sleep", true, false],
    ["wake-drop", true, false],
    ["split-slide", true, false],
    ["revolute-dd", false, false],
    ["revolute-pendulum", false, false],
    ["revolute-motor", false, false],
    ["revolute-limit", false, false],
    ["revolute-chain", true, false],
    ["weld-dd", false, false],
    ["parallel", false, false],
    ["joint-contacts", false, false],
    ["motor", false, false],
    ["motor-spring", false, false],
    ["distance", false, false],
    ["distance-spring", false, false],
    ["prismatic", false, false],
    ["prismatic-motor", false, false],
    ["spherical", false, false],
    ["spherical-limits", false, false],
    ["spherical-motor", false, false],
    ["wheel", false, false],
    ["wheel-spin", false, false],
    ["wheel-steer", false, false],
    ["ragdoll", true, false],
    // Stage 11 (CCD): continuous on.
    ["ccd-drop", false, true],
    ["ccd-bullet", false, true],
    // Stage 12 (mesh contacts): a box / sphere / capsule dropped onto a static grid-mesh floor.
    ["mesh-box", false, false],
    ["mesh-sphere", false, false],
    ["mesh-capsule", false, false],
    // Stage 12 (mesh CCD): a fast box swept onto the static mesh floor (continuous on).
    ["mesh-ccd", false, true],
    // Stage 12 (height fields): box / sphere / capsule dropped onto a static grid height field, then
    // a fast box for the height-field CCD path (continuous on).
    ["height-box", false, false],
    ["height-sphere", false, false],
    ["height-capsule", false, false],
    ["height-ccd", false, true],
    // Stage 12 (compound contacts): a box dropped onto a static compound floor built from hull /
    // capsule / sphere / mesh children (sleep off).
    ["compound-hull", false, false],
    ["compound-capsule", false, false],
    ["compound-sphere", false, false],
    ["compound-mesh", false, false],
    // Stage 12 (compound CCD): a fast box swept onto the static two-hull compound floor (continuous on).
    ["compound-ccd", false, true],
    // Stage 13 (sensors): a static sensor volume that dynamic bodies fall / sweep through — the sensor
    // must not perturb dynamics, so the body hashes match the C oracle (continuous on for the fast body).
    ["sensor", false, true],
    // Stage 14 (hardening): the benchmark scenes at reduced scale — the bit-exact contract at scale
    // (large islands, many islands, a big joint grid, kinematic contact churn, the move buffer), plus a
    // 2000-step drift gate. Sleep off; large-world spawns spheres via its stepFactory.
    ["bench-pyramid", false, false],
    ["bench-many-pyramids", false, false],
    ["bench-joint-grid", false, false],
    ["bench-washer", false, false],
    ["bench-large-world", false, false],
    // Trees: cylinder stacks on a wavy mesh ground whose libm-sinf vertices are loaded from the fixture
    // (the port cannot reproduce sinf bit-exactly); the dynamics on that fixed ground are bit-exact.
    ["bench-trees", false, false],
    // Junkyard: rock pile + a kinematic cylinder swept by setTargetTransform (bit-exact pusher pose).
    ["bench-junkyard", false, false],
    // Rain: ragdolls (14-bone articulated joint island) dropped onto a mesh ground, spawned over time.
    ["bench-rain", false, false],
    ["drift", false, false],
];

// The hashes are the C reference's, generated serially — and cross-thread-count determinism is what the
// ported task machinery guarantees (within a color no two constraints share a body, the overflow color
// and contact creation stay serial in creation order, no reduction depends on worker identity). So the
// same 52 fixtures gate the multithreaded kernel unchanged: `TUMBLE_THREADS=n bun run test:fixture:mt`,
// or `TUMBLE_THREADS=auto` to drive the default-on path — bare `init()`, which multithreads standalone.
const RAW = process.env.TUMBLE_THREADS;
const AUTO = RAW === "auto";
const THREADS = AUTO ? undefined : Number(RAW ?? 0);
// AUTO and any explicit count ≥ 1 must land on the shared kernel here (bun/node have SAB unconditionally);
// only `TUMBLE_THREADS` absent/0 stays single-thread.
const WANT_MT = AUTO || (THREADS ?? 0) >= 1;

beforeAll(async () => {
    await (AUTO ? init() : init({ threads: THREADS }));
    // A silent fall back to single-thread would make the MT run a vacuous re-run of the default one, so
    // the gate asserts the shared kernel actually loaded (`sharedBytes` is 0 on the ST path).
    if (WANT_MT && sharedBytes() === 0) {
        throw new Error(`wanted the multithreaded kernel (${RAW}), got the single-thread one`);
    }
    console.log(`[fixtures] threads: ${threads()}${sharedBytes() > 0 ? " (shared kernel)" : ""}`);
});

afterAll(async () => {
    await shutdown();
});

describe("fixture parity", () => {
    for (const [scene, enableSleep, enableContinuous] of SCENES) {
        test(scene, () => {
            const fx = loadFixture(scene);
            const timeStep = fround(fx.timeStep);

            const world = new World({
                gravity: { x: fx.gravity[0], y: fx.gravity[1], z: fx.gravity[2] },
                enableSleep,
                enableContinuous,
            });
            builders[sceneBuilder[scene] ?? scene](world, fx);
            const stepFn = stepFactories[scene]?.();

            for (let step = 0; step < fx.stepCount; ++step) {
                stepFn?.(world, step);
                world.step(timeStep, fx.subStepCount);
                const got = toHex(hashWorldState(world.state));
                if (got !== fx.hashes[step]) {
                    const dump = dumpBodies(world);
                    const ref = fx.states.find((s) => s.step === step);
                    let msg = `${scene}: hash diverged at step ${step}\n  got  ${got}\n  want ${fx.hashes[step]}\n`;
                    msg += `  port bodies: ${JSON.stringify(dump)}\n`;
                    if (ref) {
                        msg += `  ref bodies:  ${JSON.stringify(ref.bodies)}\n`;
                    } else {
                        msg += `  (no reference state dump at step ${step})\n`;
                    }
                    throw new Error(msg);
                }
                expect(got).toBe(fx.hashes[step]);
            }

            expectIdentityRecords();
            world.destroy();
        });
    }
});

// Joint events are flagged during the joint solve (getJointReaction vs threshold), so on the pool the
// solve runs in-kernel and the flags must be rebuilt from the read-back impulses — a behavioral property
// the world-state hash can't see. This lives in the fixture harness (not engine/events.test.ts) because
// the check must run at ST, t2, and t8, and the kernel is a process singleton whose thread count latches
// at the first init(): only the per-count fixture processes (test:fixture / :mt / :auto) realize all
// three. The ST run passes on the serial path pre-fix; the MT runs are the regression that was red.
describe("joint events", () => {
    // A zero-threshold weld reports its joint every awake step; a default (no-threshold) weld never does.
    // Mirrors events.test.ts's serial "joint events" specs, but under the harness's resolved thread count.
    test("a zero-threshold joint reports every awake step", () => {
        const world = new World();
        const anchor = world.createBody({ type: BodyType.Static, position: { x: 0, y: 5, z: 0 } });
        const hung = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 4, z: 0 } });
        hung.createHull({ density: 1 }, makeBoxHull(0.5, 0.5, 0.5));
        const flagged = world.createWeldJoint(anchor, hung, {
            forceThreshold: 0,
            userData: "load",
        });

        world.step(1 / 60);

        const events = world.getJointEvents();
        expect(events.length).toBe(1);
        expect(events[0].joint.id.index1).toBe(flagged.id.index1);
        expect(events[0].userData).toBe("load");

        world.destroy();
    });

    test("a default (no-threshold) joint reports nothing", () => {
        const world = new World();
        const anchor = world.createBody({ type: BodyType.Static, position: { x: 0, y: 5, z: 0 } });
        const hung = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 4, z: 0 } });
        hung.createHull({ density: 1 }, makeBoxHull(0.5, 0.5, 0.5));
        world.createWeldJoint(anchor, hung);

        world.step(1 / 60);
        expect(world.getJointEvents().length).toBe(0);

        world.destroy();
    });
});
