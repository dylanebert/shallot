import {
    Transform,
    PhysicsPlugin,
    Body,
    Force,
    Impulse,
    Velocity,
    BallJoint,
    SpringJoint,
    Part,
    Shape,
    Camera,
    Shadows,
    Orbit,
    Tonemap,
    FXAA,
    AmbientLight,
    DirectionalLight,
    RenderPlugin,
    Inputs,
    ActiveCamera,
    Mesh,
    mesh,
    createCone,
    hull,
    raycast,
    screenToRay,
} from "@dylanebert/shallot";
import type { Plugin, State, System } from "@dylanebert/shallot";
import { OutlinePlugin, Outline } from "@dylanebert/shallot/extras";
import { BenchConfig } from "../config";
import { StepCounterSystem } from "./arena";

export const PHYSICS_TEST_VARIANTS = [
    "box",
    "sphere",
    "capsule",
    "cone",
    "mixed",
    "stack",
    "pyramid",
    "rope",
    "heavy-rope",
    "spring",
    "spring-chain",
    "bridge",
    "gravity",
    "force",
    "impulse",
    "velocity",
    "filter",
] as const;
export type PhysicsTestVariant = (typeof PHYSICS_TEST_VARIANTS)[number];

let physicsTestState: State | null = null;
let physicsTestEntities: number[] = [];
let coneMeshId = -1;

let anchorEid = -1;
let grabJointEid = -1;
let dragTarget = -1;
let dragRayDistance = 0;
let dragLocalAnchor: [number, number, number] = [0, 0, 0];
let wasLeftDown = false;

function worldToLocal(
    eid: number,
    point: { x: number; y: number; z: number },
): [number, number, number] {
    const px = point.x - Transform.posX[eid];
    const py = point.y - Transform.posY[eid];
    const pz = point.z - Transform.posZ[eid];
    const qx = -Transform.quatX[eid];
    const qy = -Transform.quatY[eid];
    const qz = -Transform.quatZ[eid];
    const qw = Transform.quatW[eid];
    const tx = 2 * (qy * pz - qz * py);
    const ty = 2 * (qz * px - qx * pz);
    const tz = 2 * (qx * py - qy * px);
    return [
        px + qw * tx + qy * tz - qz * ty,
        py + qw * ty + qz * tx - qx * tz,
        pz + qw * tz + qx * ty - qy * tx,
    ];
}

function addMeshBody(
    state: State,
    meshId: number,
    x: number,
    y: number,
    z: number,
    sx: number,
    sy: number,
    sz: number,
    mass: number,
    friction: number,
    color: number,
): number {
    const eid = addBody(state, Shape.Mesh, x, y, z, sx, sy, sz, mass, friction, color);
    state.addComponent(eid, Mesh);
    Mesh.geometry[eid] = meshId;
    return eid;
}

function addBody(
    state: State,
    shape: number,
    x: number,
    y: number,
    z: number,
    sx: number,
    sy: number,
    sz: number,
    mass: number,
    friction: number,
    color: number,
): number {
    const eid = state.addEntity();
    state.addComponent(eid, Transform);
    state.addComponent(eid, Part);
    state.addComponent(eid, Body);
    Transform.posX[eid] = x;
    Transform.posY[eid] = y;
    Transform.posZ[eid] = z;
    Part.shape[eid] = shape;
    Part.sizeX[eid] = sx;
    Part.sizeY[eid] = sy;
    Part.sizeZ[eid] = sz;
    Part.color[eid] = color;
    Body.mass[eid] = mass;
    Body.friction[eid] = friction;
    physicsTestEntities.push(eid);
    return eid;
}

function addBallJoint(
    state: State,
    bodyA: number,
    bodyB: number,
    anchorA: [number, number, number],
    anchorB: [number, number, number] = [0, 0, 0],
): number {
    const eid = state.addEntity();
    state.addComponent(eid, BallJoint);
    BallJoint.bodyA[eid] = bodyA;
    BallJoint.bodyB[eid] = bodyB;
    BallJoint.anchorAX[eid] = anchorA[0];
    BallJoint.anchorAY[eid] = anchorA[1];
    BallJoint.anchorAZ[eid] = anchorA[2];
    BallJoint.anchorBX[eid] = anchorB[0];
    BallJoint.anchorBY[eid] = anchorB[1];
    BallJoint.anchorBZ[eid] = anchorB[2];
    physicsTestEntities.push(eid);
    return eid;
}

function addSpringJoint(
    state: State,
    bodyA: number,
    bodyB: number,
    anchorA: [number, number, number],
    anchorB: [number, number, number],
    restLength: number,
    stiffness: number,
): number {
    const eid = state.addEntity();
    state.addComponent(eid, SpringJoint);
    SpringJoint.bodyA[eid] = bodyA;
    SpringJoint.bodyB[eid] = bodyB;
    SpringJoint.anchorAX[eid] = anchorA[0];
    SpringJoint.anchorAY[eid] = anchorA[1];
    SpringJoint.anchorAZ[eid] = anchorA[2];
    SpringJoint.anchorBX[eid] = anchorB[0];
    SpringJoint.anchorBY[eid] = anchorB[1];
    SpringJoint.anchorBZ[eid] = anchorB[2];
    SpringJoint.restLength[eid] = restLength;
    SpringJoint.stiffness[eid] = stiffness;
    physicsTestEntities.push(eid);
    return eid;
}

function makeGround(state: State, y = 0, friction = 0.5): number {
    return addBody(state, Shape.Box, 0, y, 0, 100, 1, 100, 0, friction, 0x252220);
}

function buildBox(state: State) {
    addBody(state, Shape.Box, 0, 4, 0, 1, 1, 1, 1, 0.5, 0xd49560);
    makeGround(state);
}

function buildStack(state: State) {
    for (let i = 9; i >= 0; i--) {
        addBody(state, Shape.Box, 0, i * 1.5 + 1.0, 0, 1, 1, 1, 1, 0.5, 0xd49560);
    }
    makeGround(state);
}

function buildPyramid(state: State) {
    const size = 16;
    for (let y = size - 1; y >= 0; y--) {
        for (let x = size - y - 1; x >= 0; x--) {
            addBody(
                state,
                Shape.Box,
                x * 1.01 + y * 0.5 - size / 2.0,
                y * 0.51 + 0.26,
                0,
                1,
                0.5,
                0.5,
                0.25,
                0.5,
                0xd49560,
            );
        }
    }
    makeGround(state, -0.5);
}

function buildRope(state: State) {
    const links: number[] = new Array(20);
    for (let i = 19; i >= 0; i--) {
        links[i] = addBody(
            state,
            Shape.Box,
            i,
            10,
            0,
            1,
            0.5,
            0.5,
            i === 0 ? 0 : 0.25,
            0.5,
            i === 0 ? 0x888888 : 0xd49560,
        );
    }
    makeGround(state, -20);
    for (let i = 1; i < links.length; i++) {
        addBallJoint(state, links[i - 1], links[i], [0.5, 0, 0], [-0.5, 0, 0]);
    }
}

function buildHeavyRope(state: State) {
    const N = 20;
    const Size = 5;
    const links: number[] = new Array(N);
    for (let i = N - 1; i >= 0; i--) {
        const isLast = i === N - 1;
        const s = isLast ? Size : 1;
        const sy = isLast ? Size : 0.5;
        const sz = isLast ? Size : 0.5;
        const x = i + (isLast ? Size / 2 : 0);
        const mass = i === 0 ? 0 : s * sy * sz;
        links[i] = addBody(
            state,
            Shape.Box,
            x,
            10,
            0,
            s,
            sy,
            sz,
            mass,
            0.5,
            i === 0 ? 0x888888 : isLast ? 0x4078a0 : 0xd49560,
        );
    }
    makeGround(state, -20);
    for (let i = 1; i < links.length; i++) {
        const isLast = i === N - 1;
        addBallJoint(state, links[i - 1], links[i], [0.5, 0, 0], [isLast ? -Size / 2 : -0.5, 0, 0]);
    }
}

function buildSpring(state: State) {
    const block = addBody(state, Shape.Box, 0, 8, 0, 2, 2, 2, 8, 0.5, 0x6090b0);
    const anchor = addBody(state, Shape.Box, 0, 14, 0, 1, 1, 1, 0, 0.5, 0x888888);
    makeGround(state);
    addSpringJoint(state, anchor, block, [0, 0, 0], [0, 0, 0], 4.0, 100);
}

function buildSpringChain(state: State) {
    const N = 8;
    const links: number[] = new Array(N);
    for (let i = N - 1; i >= 0; i--) {
        const x = (i - (N - 1) * 0.5) * 3.0;
        const mass = i === 0 || i === N - 1 ? 0 : 0.5625;
        links[i] = addBody(
            state,
            Shape.Box,
            x,
            12,
            0,
            1,
            0.75,
            0.75,
            mass,
            0.5,
            i === 0 || i === N - 1 ? 0x888888 : 0xd49560,
        );
    }
    makeGround(state, -10);
    for (let i = 1; i < links.length; i++) {
        addSpringJoint(
            state,
            links[i - 1],
            links[i],
            [0.5, 0, 0],
            [-0.5, 0, 0],
            3.0,
            i % 2 === 0 ? 10.0 : 10000.0,
        );
    }
}

function buildBridge(state: State) {
    const N = 40;
    const plankLength = 1.0;
    const plankWidth = 4.0;
    const plankHeight = 0.5;
    const halfLength = plankLength * 0.5;
    const halfWidth = plankWidth * 0.5;

    const planks: number[] = new Array(N);
    for (let i = N - 1; i >= 0; i--) {
        const mass = i === 0 || i === N - 1 ? 0 : plankLength * plankHeight * plankWidth;
        planks[i] = addBody(
            state,
            Shape.Box,
            i - N / 2.0,
            10,
            0,
            plankLength,
            plankHeight,
            plankWidth,
            mass,
            0.5,
            i === 0 || i === N - 1 ? 0x888888 : 0xd49560,
        );
    }
    makeGround(state);
    for (let i = 1; i < planks.length; i++) {
        addBallJoint(
            state,
            planks[i - 1],
            planks[i],
            [halfLength, 0, halfWidth],
            [-halfLength, 0, halfWidth],
        );
        addBallJoint(
            state,
            planks[i - 1],
            planks[i],
            [halfLength, 0, -halfWidth],
            [-halfLength, 0, -halfWidth],
        );
    }

    for (let x = N / 4 - 1; x >= 0; x--) {
        for (let y = N / 8 - 1; y >= 0; y--) {
            addBody(state, Shape.Box, x - N / 8.0, y + 12.0, 0, 1, 1, 1, 1, 0.5, 0x4078a0);
        }
    }
}

function buildMixed(state: State) {
    addBody(state, Shape.Box, 0, -0.5, 0, 20, 1, 20, 0, 0.5, 0x252220);
    for (const b of [
        { shape: Shape.Sphere, x: 0, y: 8, z: 0.5, size: 1, color: 0x8b6040 },
        { shape: Shape.Sphere, x: 0.8, y: 12, z: -0.4, size: 1.6, color: 0xa49070 },
        { shape: Shape.Box, x: -0.6, y: 6, z: 0.3, size: 1, color: 0xd49560 },
        { shape: Shape.Capsule, x: 0.3, y: 15, z: -0.8, size: 1.2, color: 0x4078a0 },
        { shape: Shape.Sphere, x: -0.5, y: 10, z: 1, size: 0.8, color: 0x806848 },
        { shape: Shape.Box, x: 1, y: 20, z: -0.2, size: 1.4, color: 0xd49560 },
        { shape: Shape.Capsule, x: -1, y: 18, z: 0.6, size: 1, color: 0x6090b0 },
        { shape: Shape.Sphere, x: 0.5, y: 25, z: -0.5, size: 0.6, color: 0xe8e0d8 },
    ]) {
        addBody(state, b.shape, b.x, b.y, b.z, b.size, b.size, b.size, 1, 0.5, b.color);
    }
    addMeshBody(state, coneMeshId, 0.5, 10, 0.5, 1.2, 1.5, 1.2, 2, 0.5, 0xa49070);
}

function buildSphere(state: State) {
    addBody(state, Shape.Sphere, 0, 4, 0, 1, 1, 1, 1, 0.5, 0xa49070);
    makeGround(state);
}

function buildCapsule(state: State) {
    addBody(state, Shape.Capsule, 0, 4, 0, 1, 1, 1, 1, 0.5, 0xa49070);
    makeGround(state);
}

function buildGravity(state: State) {
    addBody(state, Shape.Box, -2, 4, 0, 1, 1, 1, 1, 0.5, 0xd49560);
    const floating = addBody(state, Shape.Box, 2, 4, 0, 1, 1, 1, 1, 0.5, 0x4078a0);
    Body.gravity[floating] = 0;
    makeGround(state);
}

function buildForce(state: State) {
    const box = addBody(state, Shape.Box, 0, 4, 0, 1, 1, 1, 1, 0.5, 0x6090b0);
    state.addComponent(box, Force);
    Force.forceY[box] = 8;
    Force.forceX[box] = 1;
    Force.forceZ[box] = -0.5;
    makeGround(state);
}

let impulseBoxEid = -1;
let lastImpulseTime = 0;

function buildImpulse(state: State) {
    impulseBoxEid = addBody(state, Shape.Box, 0, 1, 0, 1, 1, 1, 1, 0.5, 0x806890);
    lastImpulseTime = 0;
    makeGround(state);
}

const IMPULSE_INTERVAL = 2;

const ImpulseSystem: System = {
    group: "fixed",
    update(state) {
        if (impulseBoxEid < 0) return;
        const t = state.time.elapsed;
        if (t - lastImpulseTime < IMPULSE_INTERVAL) return;
        lastImpulseTime = t;
        if (!state.hasComponent(impulseBoxEid, Body)) return;
        state.addComponent(impulseBoxEid, Impulse);
        Impulse.impulseY[impulseBoxEid] = 12;
        Impulse.angularImpulseX[impulseBoxEid] = 2;
        Impulse.angularImpulseZ[impulseBoxEid] = -1;
    },
};

let velocityBoxEid = -1;
let lastVelocityTime = 0;

function buildVelocity(state: State) {
    velocityBoxEid = addBody(state, Shape.Box, 0, 1, 0, 1, 1, 1, 1, 0.5, 0x6090b0);
    Body.gravity[velocityBoxEid] = 3;
    lastVelocityTime = 0;
    makeGround(state);
}

const VELOCITY_INTERVAL = 2;

const VelocitySystem: System = {
    group: "fixed",
    update(state) {
        if (velocityBoxEid < 0) return;
        const t = state.time.elapsed;
        if (t - lastVelocityTime < VELOCITY_INTERVAL) return;
        lastVelocityTime = t;
        if (!state.hasComponent(velocityBoxEid, Body)) return;
        state.addComponent(velocityBoxEid, Velocity);
        Velocity.linearY[velocityBoxEid] = 10;
    },
};

function buildCone(state: State) {
    addMeshBody(state, coneMeshId, 0, 4, 0, 1, 1, 1, 1, 0.5, 0xa49070);
    makeGround(state);
}

function buildFilter(state: State) {
    for (let i = 0; i < 3; i++) {
        const eid = addBody(
            state,
            Shape.Box,
            i * 1.5 - 1.5,
            6 + i * 2,
            0,
            1,
            1,
            1,
            1,
            0.5,
            0xa05050,
        );
        Body.group[eid] = 1;
    }
    for (let i = 0; i < 3; i++) {
        const eid = addBody(
            state,
            Shape.Box,
            i * 1.5 - 1.5,
            12 + i * 2,
            0,
            1,
            1,
            1,
            1,
            0.5,
            0x4078a0,
        );
        Body.group[eid] = 2;
    }
    makeGround(state);
}

const PHYSICS_TESTS: Record<PhysicsTestVariant, (state: State) => void> = {
    box: buildBox,
    sphere: buildSphere,
    capsule: buildCapsule,
    cone: buildCone,
    mixed: buildMixed,
    stack: buildStack,
    pyramid: buildPyramid,
    rope: buildRope,
    "heavy-rope": buildHeavyRope,
    spring: buildSpring,
    "spring-chain": buildSpringChain,
    bridge: buildBridge,
    gravity: buildGravity,
    force: buildForce,
    impulse: buildImpulse,
    velocity: buildVelocity,
    filter: buildFilter,
};

export function setPhysicsTestVariant(name: PhysicsTestVariant) {
    const state = physicsTestState;
    if (!state) return;

    if (grabJointEid >= 0) {
        state.removeEntity(grabJointEid);
        grabJointEid = -1;
        const cam = ActiveCamera.from(state);
        if (cam && cam.eid >= 0) Orbit.suppress[cam.eid] = 0;
    }
    dragTarget = -1;
    wasLeftDown = false;

    for (const eid of physicsTestEntities) state.removeEntity(eid);
    physicsTestEntities = [];
    impulseBoxEid = -1;

    const builder = PHYSICS_TESTS[name];
    if (!builder) {
        const valid = Object.keys(PHYSICS_TESTS).join(", ");
        throw new Error(`unknown physics test "${name}". valid: ${valid}`);
    }
    builder(state);
}

let hoveredEid = -1;

const GrabSystem: System = {
    setup(state) {
        state.setResource(Outline, {
            getEntities: () => (hoveredEid >= 0 ? [hoveredEid] : []),
            color: 0xffffff,
            thickness: 2,
        });
    },
    update(state) {
        const input = Inputs.from(state);
        const cam = ActiveCamera.from(state);
        if (!input || !cam || cam.eid < 0) {
            hoveredEid = -1;
            return;
        }

        const mouse = input.mouse;
        const leftDown = mouse.left;
        const leftPressed = leftDown && !wasLeftDown;
        const leftReleased = !leftDown && wasLeftDown;
        wasLeftDown = leftDown;

        if (!mouse.hover && !leftDown) {
            hoveredEid = -1;
            return;
        }

        const ray = screenToRay(
            state,
            mouse.x / mouse.canvasWidth,
            mouse.y / mouse.canvasHeight,
            mouse.canvasWidth,
            mouse.canvasHeight,
        );
        if (!ray) return;

        if (leftPressed && dragTarget < 0) {
            const entities = state.query([Part, Transform, Body]);
            const hit = raycast(entities, ray);
            if (hit && Body.mass[hit.eid] > 0) {
                dragTarget = hit.eid;
                dragLocalAnchor = worldToLocal(hit.eid, hit.point);
                const ox = ray.origin.x;
                const oy = ray.origin.y;
                const oz = ray.origin.z;
                dragRayDistance = Math.max(
                    (hit.point.x - ox) * ray.direction.x +
                        (hit.point.y - oy) * ray.direction.y +
                        (hit.point.z - oz) * ray.direction.z,
                    0.1,
                );
                Transform.posX[anchorEid] = hit.point.x;
                Transform.posY[anchorEid] = hit.point.y;
                Transform.posZ[anchorEid] = hit.point.z;

                grabJointEid = state.addEntity();
                state.addComponent(grabJointEid, BallJoint);
                BallJoint.bodyA[grabJointEid] = anchorEid;
                BallJoint.bodyB[grabJointEid] = dragTarget;
                BallJoint.anchorBX[grabJointEid] = dragLocalAnchor[0];
                BallJoint.anchorBY[grabJointEid] = dragLocalAnchor[1];
                BallJoint.anchorBZ[grabJointEid] = dragLocalAnchor[2];
                BallJoint.stiffness[grabJointEid] = 5000;
                Orbit.suppress[cam.eid] = 1;
            }
        }

        if (dragTarget >= 0 && leftDown) {
            Transform.posX[anchorEid] = ray.origin.x + ray.direction.x * dragRayDistance;
            Transform.posY[anchorEid] = ray.origin.y + ray.direction.y * dragRayDistance;
            Transform.posZ[anchorEid] = ray.origin.z + ray.direction.z * dragRayDistance;
        }

        if (leftReleased && dragTarget >= 0) {
            dragTarget = -1;
            Transform.posY[anchorEid] = -1000;
            state.removeEntity(grabJointEid);
            grabJointEid = -1;
            Orbit.suppress[cam.eid] = 0;
        }

        if (dragTarget < 0) {
            const entities = state.query([Part, Transform, Body]);
            const hit = raycast(entities, ray);
            hoveredEid = hit && Body.mass[hit.eid] > 0 ? hit.eid : -1;
        }
    },
};

export function buildPhysicsTestPlugin(testName: PhysicsTestVariant): Plugin {
    return {
        name: "PhysicsTest",
        dependencies: [PhysicsPlugin, RenderPlugin, OutlinePlugin],
        systems: [StepCounterSystem, GrabSystem, ImpulseSystem, VelocitySystem],
        initialize(state) {
            physicsTestState = state;
            coneMeshId = mesh(createCone(), "cone");
            hull(coneMeshId);

            const cam = state.addEntity();
            state.addComponent(cam, Transform);
            state.addComponent(cam, Camera);
            state.addComponent(cam, Tonemap);
            state.addComponent(cam, FXAA);
            state.addComponent(cam, Shadows);
            state.addComponent(cam, Orbit);
            Orbit.distance[cam] = 20;
            Orbit.maxDistance[cam] = 100;
            Orbit.pitch[cam] = Math.PI / 8;
            state.addComponent(cam, BenchConfig);

            anchorEid = state.addEntity();
            state.addComponent(anchorEid, Transform);
            state.addComponent(anchorEid, Part);
            state.addComponent(anchorEid, Body);
            Part.shape[anchorEid] = Shape.Sphere;
            Part.sizeX[anchorEid] = 0.01;
            Part.sizeY[anchorEid] = 0.01;
            Part.sizeZ[anchorEid] = 0.01;
            Body.mass[anchorEid] = 0;
            Transform.posY[anchorEid] = -1000;

            const ambient = state.addEntity();
            state.addComponent(ambient, AmbientLight);
            const dir = state.addEntity();
            state.addComponent(dir, DirectionalLight);

            setPhysicsTestVariant(testName);
        },
    };
}
