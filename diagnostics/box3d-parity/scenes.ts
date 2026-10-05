// Shallot side of the box3d-parity divergence oracle (divergence.oracle.ts): native.c's scenes through the
// physics API, stepped and printed the same way, so the two outputs diff line for line.
//
//   bun diagnostics/box3d-parity/scenes.ts <scene> <threads> <steps>
//
// The builders transcribe Box3D 47d7f7cc's shared/benchmarks.c and shared/human.c with their f32
// arithmetic; human.c's bone table is read from the frozen ragdoll fixture. Environment as native.c:
// RAIN_COUNT, RAIN_GROUP, ROCKS, COLORS, CACHE, PROFILE, and on Node CPU (cpu.ts).
import { World } from "../../src/engine";
import {
    type Body,
    BodyType,
    createCylinder,
    createGridMesh,
    createRock,
    createTorusMesh,
    defaultFilter,
    defaultSurfaceMaterial,
    hash,
    init,
    type Joint,
    makeBoxHull,
    makeOffsetBoxHull,
    PhysicsWorld,
    shutdown,
} from "../../src/standard/physics/api";
import { DIR_STRIDE } from "../../src/standard/physics/collision/manifoldstore";
import {
    computeCosSin,
    DEG_TO_RAD,
    offsetPos,
    PI,
    quat,
} from "../../src/standard/physics/common/math";
import BONE_TABLE from "../../src/standard/physics/solver/fixtures/human.json";
import { PROFILE_FIELDS } from "../../src/standard/physics/world/profile";
import type { WorldState } from "../../src/standard/physics/world/world";

const f = Math.fround;
const hex = (v: number) => (v >>> 0).toString(16).padStart(8, "0");
const env = (name: string, fallback: number) => Number(process.env[name] ?? fallback);

type Scene = {
    def: object;
    create: (w: PhysicsWorld) => void;
    step: (w: PhysicsWorld, i: number) => void;
};

// shared/human.c CreateHuman and DestroyHuman.
type V3 = [number, number, number];
type Q4 = [number, number, number, number];
type Bone = {
    parent: number;
    refP: V3;
    refQ: Q4;
    cap1: V3;
    cap2: V3;
    radius: number;
    negGroup: boolean;
    joint?: "spherical" | "revolute";
    lfaP?: V3;
    lfaQ?: Q4;
    lfbP?: V3;
    lfbQ?: Q4;
    swingDeg?: number;
    twistDeg?: [number, number];
    friction: number;
};
const BONES = BONE_TABLE as Bone[];
type Human = { bodies: Body[]; joints: Joint[]; filter: Joint };

function createHuman(
    w: PhysicsWorld,
    position: { x: number; y: number; z: number },
    groupIndex: number,
): Human {
    const frictionTorque = 5,
        hertz = 1,
        dampingRatio = f(0.7);
    const bodies = BONES.map((b) => {
        const body = w.createBody({
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
        return body;
    });
    const frame = (p: V3, q: Q4) => ({
        p: { x: f(p[0]), y: f(p[1]), z: f(p[2]) },
        q: quat.normalize({ v: { x: f(q[0]), y: f(q[1]), z: f(q[2]) }, s: f(q[3]) }),
    });
    const joints: Joint[] = [];
    for (let i = 1; i < BONES.length; ++i) {
        const b = BONES[i];
        const twist = b.twistDeg as [number, number];
        const base = {
            localFrameA: frame(b.lfaP as V3, b.lfaQ as Q4),
            localFrameB: frame(b.lfbP as V3, b.lfbQ as Q4),
            enableSpring: hertz > 0,
            hertz,
            dampingRatio,
            enableMotor: true,
            maxMotorTorque: f(f(b.friction) * frictionTorque),
        };
        joints.push(
            b.joint === "revolute"
                ? w.createRevoluteJoint(bodies[b.parent], bodies[i], {
                      ...base,
                      enableLimit: true,
                      lowerAngle: f(twist[0] * DEG_TO_RAD),
                      upperAngle: f(twist[1] * DEG_TO_RAD),
                  })
                : w.createSphericalJoint(bodies[b.parent], bodies[i], {
                      ...base,
                      enableConeLimit: true,
                      coneAngle: f((b.swingDeg as number) * DEG_TO_RAD),
                      enableTwistLimit: true,
                      lowerTwistAngle: f(twist[0] * DEG_TO_RAD),
                      upperTwistAngle: f(twist[1] * DEG_TO_RAD),
                  }),
        );
    }
    return { bodies, joints, filter: w.createFilterJoint(bodies[6], bodies[8]) };
}

function destroyHuman(h: Human): void {
    h.filter.destroy(false);
    for (const j of h.joints) j.destroy(false);
    for (const b of h.bodies) b.destroy();
}

// benchmarks.c CreateRain/StepRain; rain is rain-n at RAIN_COUNT 10, RAIN_GROUP 3.
function rain(count: number, group: number): Scene {
    const size = 15;
    const groups: Human[][] = [];
    let columnCount = 0,
        columnIndex = 0;
    const createGroup = (w: PhysicsWorld, row: number, column: number) => {
        const groupIndex = row * count + column;
        const span = f(count * size),
            distance = f(f(1.0 * span) / count);
        const position = {
            x: f(f(-0.5 * span) + f(distance * f(column + 0.5))),
            y: 20,
            z: f(f(-0.5 * span) + f(distance * f(row + 0.5))),
        };
        groups[groupIndex] = [];
        for (let i = 0; i < group; ++i) {
            groups[groupIndex].push(createHuman(w, { ...position }, groupIndex));
            position.x = f(position.x + 0.75);
        }
    };
    return {
        def: {},
        create(w) {
            const grid = createGridMesh(8, 8, f(size / f(2 * 4)), 1, true);
            const torus = createTorusMesh(16, 16, f(0.25 * size), 1.0);
            const span = f(size * count);
            let x = f(f(-0.5 * span) + f(0.5 * size));
            for (let i = 0; i < count; ++i) {
                let z = f(f(-0.5 * span) + f(0.5 * size));
                for (let j = 0; j < count; ++j) {
                    const body = w.createBody({ position: { x, y: 0, z } });
                    body.createMesh({}, grid, { x: 1, y: 1, z: 1 });
                    body.createMesh({}, torus, { x: 1, y: 1, z: 1 });
                    z = f(z + size);
                }
                x = f(x + size);
            }
        },
        step(w, stepCount) {
            if ((stepCount & 0x2f) !== 0) return;
            if (columnCount < count) {
                for (let i = 0; i < count; ++i) createGroup(w, i, columnCount);
                columnCount = Math.min(columnCount + 1, count);
                return;
            }
            for (let i = 0; i < count; ++i) {
                for (const h of groups[i * count + columnIndex]) destroyHuman(h);
                createGroup(w, i, columnIndex);
            }
            columnIndex = columnIndex + 1 >= count ? 0 : columnIndex + 1;
        },
    };
}

// benchmarks.c CreateJunkyard/StepJunkyard; junkyard keeps every rock of the 24 x 21 x 21 grid.
function junk(rocks: V3[], def: object): Scene {
    let pusher: Body;
    let degrees = 0;
    return {
        def,
        create(w) {
            const ground = w.createBody({ position: { x: 0, y: -1, z: 0 } });
            ground.createHull({}, makeBoxHull(120, 1, 120));
            ground.createHull({}, makeOffsetBoxHull(1, 8, 50, { x: -50, y: 8, z: 0 }));
            ground.createHull({}, makeOffsetBoxHull(1, 8, 50, { x: 50, y: 8, z: 0 }));
            ground.createHull({}, makeOffsetBoxHull(50, 8, 1, { x: 0, y: 8, z: -50 }));
            ground.createHull({}, makeOffsetBoxHull(50, 8, 1, { x: 0, y: 8, z: 50 }));
            const rock = createRock(1.5);
            for (const [X, Y, Z] of rocks) {
                const position = {
                    x: f(-40 + f(4 * X)),
                    y: f(f(f(4 * Y) + 24) + 1),
                    z: f(-40 + f(4 * Z)),
                };
                w.createBody({ type: BodyType.Dynamic, position }).createHull({}, rock);
            }
            pusher = w.createBody({ type: BodyType.Kinematic, position: { x: 35, y: 0, z: 0 } });
            pusher.createHull({}, createCylinder(24, 4, 0, 16));
        },
        step() {
            const timeStep = f(1 / 60);
            degrees = f(degrees + f(-6.0 * timeStep));
            const cs = computeCosSin(f(f(degrees * PI) / 180.0));
            const p = { x: f(35 * cs.cosine), y: 0, z: f(35 * cs.sine) };
            pusher.setTargetTransform({ p, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } }, timeStep, false);
        },
    };
}

// benchmarks.c CreateJointGrid.
function jointGrid(): Scene {
    const frame = (x: number, y: number, z: number) => ({
        p: { x, y, z },
        q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
    });
    return {
        def: { enableSleep: false },
        create(w) {
            const n = 100;
            const bodies: Body[] = [];
            const filter = { categoryBits: 2n, maskBits: 0xffffffffn ^ 2n, groupIndex: 0 };
            const sphere = { center: { x: 0, y: 0, z: 0 }, radius: f(0.4) };
            for (let k = 0; k < n; ++k) {
                for (let i = 0; i < n; ++i) {
                    const body = w.createBody({
                        type: i === 0 ? BodyType.Static : BodyType.Dynamic,
                        enableSleep: false,
                        position: { x: k, y: -i, z: 0 },
                    });
                    body.createSphere({ filter }, sphere);
                    const index = bodies.length;
                    if (i > 0)
                        w.createSphericalJoint(bodies[index - 1], body, {
                            localFrameA: frame(0, -0.5, 0),
                            localFrameB: frame(0, 0.5, 0),
                        });
                    if (k > 0)
                        w.createSphericalJoint(bodies[index - n], body, {
                            localFrameA: frame(0.5, 0, 0),
                            localFrameB: frame(-0.5, 0, 0),
                        });
                    bodies.push(body);
                }
            }
        },
        step() {},
    };
}

const all: V3[] = [];
for (let Y = 0; Y < 24; ++Y)
    for (let X = 0; X <= 20; ++X) for (let Z = 0; Z <= 20; ++Z) all.push([X, Y, Z]);
const chosen = (process.env.ROCKS ?? "")
    .split(";")
    .filter(Boolean)
    .map((t) => t.split(",").map(Number) as V3);
const junkyardCapacity = {
    capacity: {
        staticShapeCount: 16,
        dynamicShapeCount: 20 * 20 * 24 + 1,
        staticBodyCount: 16,
        dynamicBodyCount: 20 * 20 * 24 + 1,
        contactCount: 250 * 1024,
    },
};

const [name, threadArg, stepArg] = process.argv.slice(2);
const scenes: Record<string, () => Scene> = {
    rain: () => rain(10, 3),
    "rain-n": () => rain(env("RAIN_COUNT", 10), env("RAIN_GROUP", 3)),
    junkyard: () => junk(all, junkyardCapacity),
    junk: () => junk(chosen, {}),
    // biome-ignore lint/style/useNamingConvention: Box3D's benchmark name, as native.c takes it.
    joint_grid: jointGrid,
};
if (!scenes[name] || stepArg === undefined) {
    console.error("usage: scenes.ts rain|rain-n|junkyard|junk|joint_grid <threads> <steps>");
    process.exit(2);
}
const owner = new World();
const threads = Number(threadArg);
await init(owner, { threads: threads === 1 ? 0 : threads });
const scene = scenes[name]();
const w = new PhysicsWorld({ enableContinuous: true, ...scene.def }, owner);
scene.create(w);
const state = w.state as WorldState;
const colors = env("COLORS", -1),
    cache = env("CACHE", -1),
    profileFrom = env("PROFILE", -1),
    cpuFrom = env("CPU", -1),
    steps = Number(stepArg);
const cpu = cpuFrom >= 0 ? await import("./cpu") : null;
const lines: string[] = [];
for (let i = 0; i < steps; ++i) {
    scene.step(w, i);
    if (i === cpuFrom) cpu?.startCpu();
    w.step(f(1 / 60), 4);
    lines.push(`${i} 0x${hash(w).toString(16).padStart(16, "0")}`);
    if (profileFrom >= 0 && i >= profileFrom) {
        const p = w.getProfile();
        lines.push(`F ${i} ${PROFILE_FIELDS.map((k) => p[k].toFixed(4)).join(" ")}`);
        const c = w.getCounters();
        lines.push(`W ${i} contacts ${c.contactCount} awake ${state.awakeContacts.length} joints ${c.jointCount}`);
    }
    if (i === colors) {
        const graph = state.constraintGraph.colors;
        for (let c = 0; c < graph.length - 1; ++c) {
            const set = graph[c].bodySet;
            for (let k = 0; k < set.blockCount * 32; ++k) {
                if (set.bits[k >>> 5] & (1 << (k & 31))) lines.push(`C ${i} color ${c} body ${k}`);
            }
        }
    }
    const contact = state.contacts[cache];
    if (contact !== undefined && contact.contactId === cache) {
        const d = state.manifoldStore.dirU,
            o = cache * DIR_STRIDE + 12;
        lines.push(
            `S ${i} contact ${cache} bodies ${contact.edges[0].bodyId} ${contact.edges[1].bodyId} manifolds ${contact.manifoldCount} cache sep ${hex(d[o])} type ${d[o + 1]} indexA ${d[o + 2]} indexB ${d[o + 3]} hit ${d[o + 4]}`,
        );
    }
}
if (cpu && cpuFrom < steps) lines.push(...cpu.stopCpu(steps - cpuFrom, threads !== 1));
console.log(lines.join("\n"));
w.destroy();
await shutdown(owner);
