import { Body, ShapeKind } from "../../core/physics";
import {
    component,
    f32 as field,
    type Plugin,
    type System,
    Time,
    u32,
    vec4,
    type World,
} from "../../engine";
import type { PhysicsWorld } from "./api/world";
import { type CollisionPlane, clipVector, solvePlanes } from "./collision/mover";
import { f32, mat3, PI, quat, vec3, xf } from "./common/math";
import { BodyType } from "./common/types";
import {
    readSimCenter,
    readSimInvInertiaWorld,
    readStateAngularVelocity,
    readStateLinearVelocity,
    simInvMass,
} from "./kernel/bodycolumns";
import { BodyField, bodyField, setBodyField } from "./kernel/bodyrecords";
import { bodyType, shapeBodyId } from "./kernel/filtercolumns";
import { queryColumns } from "./kernel/querycolumns";
import {
    physicsWorld,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    SyncPhysicsConstraintsSystem,
} from "./runtime";
import { bodyApplyLinearImpulse, getBodySim, getBodyState } from "./world/body";

export const GroundState = { InAir: 0, OnGround: 1, OnSteepGround: 2 } as const;

/** A kinematic capsule Body. Write velocity in a fixed system ordered before `CharacterPlugin.systems`; movement runs after body synchronization and before the rigid solver. Resolved velocity is written back, without gravity or input policy. Ground velocity is reported, never added to motion. */
export const Character = component(
    "Character",
    {
        /** World-space metres per second in xyz; w is unused. */
        velocity: vec4,
        groundState: u32,
        /** World-space unit normal at the pogo ray hit, or zero in air. */
        groundNormal: vec4,
        /** Hit body's velocity at the ground point, in world-space metres per second, or zero in air. */
        groundVelocity: vec4,
        /** Maximum walkable slope in radians, measured from up. */
        maxSlope: field,
        /** Unit world-space up direction. */
        up: vec4,
        /** Pogo spring speed in metres per second; persists between fixed ticks. */
        pogoVelocity: field,
    },
    {
        requires: [Body],
        defaults: () => ({
            velocity: [0, 0, 0, 0],
            groundState: GroundState.InAir,
            groundNormal: [0, 0, 0, 0],
            groundVelocity: [0, 0, 0, 0],
            maxSlope: Math.PI / 4,
            up: [0, 1, 0, 0],
            pogoVelocity: 0,
        }),
    },
);

const terms = [Character, Body];

export const characterScratch = {
    create: () => ({
        planes: Array.from(
            { length: 8 },
            (): CollisionPlane => ({
                plane: { normal: vec3.zero(), offset: 0 },
                pushLimit: 3.4028234663852886e38,
                push: 0,
                clipVelocity: true,
            }),
        ),
        points: Array.from({ length: 8 }, () => vec3.zero()),
        shapes: new Int32Array(8),
        impulses: Array.from({ length: 8 }, () => vec3.zero()),
        impulseBodies: new Int32Array(8),
        impulseCount: 0,
        pose: xf.identity(),
        start: vec3.zero(),
        velocity: vec3.zero(),
        up: vec3.zero(),
        center1: vec3.zero(),
        center2: vec3.zero(),
        local: vec3.zero(),
        ray: vec3.zero(),
        translation: vec3.zero(),
        normal: vec3.zero(),
        point: vec3.zero(),
        target: vec3.zero(),
        delta: vec3.zero(),
        groundVelocity: vec3.zero(),
        center: vec3.zero(),
        angular: vec3.zero(),
        linear: vec3.zero(),
        r: vec3.zero(),
        rn: vec3.zero(),
        mrn: vec3.zero(),
        inertia: mat3.zero(),
        vr: vec3.zero(),
        impulse: vec3.zero(),
        solved: { delta: vec3.zero(), iterationCount: 0 },
        dt: 0,
        count: 0,
        radius: 0,
        groundShape: -1,
        passes: 0,
        solverIterations: 0,
        zeroVelocity: vec3.zero(),
    }),
};

function moveCharacter(world: World, eid: number): void {
    const physics = physicsWorld(world);
    const handle = physics?.getBody(eid);
    if (!physics || !handle) return;
    const body = world.storage(Body);
    if (
        body.shape.column[eid] !== ShapeKind.Capsule ||
        body.type.column[eid] !== BodyType.Kinematic
    )
        throw new Error(`Character entity ${eid} requires a kinematic capsule Body`);
    const character = world.storage(Character);
    const s = world.resource(characterScratch);
    s.dt = Time.FIXED_DT;
    // Pose-driven kinematics must stay in the solver's live publish set, without periodic wake/clone churn.
    handle.setAwake(true);
    setBodyField(physics.state, handle.id.index1 - 1, BodyField.sleepTime, 0);
    handle.getTransform(s.pose);
    const position = s.pose.p;
    s.start.x = position.x;
    s.start.y = position.y;
    s.start.z = position.z;
    const velocity = s.velocity;
    const offset = eid * 4;
    const vc = character.velocity.column;
    velocity.x = vc[offset];
    velocity.y = vc[offset + 1];
    velocity.z = vc[offset + 2];
    const up = s.up;
    const uc = character.up.column;
    up.x = uc[offset];
    up.y = uc[offset + 1];
    up.z = uc[offset + 2];
    const extents = body.halfExtents.column;
    s.radius = extents[offset + 3];
    const half = extents[offset + 1];
    s.local.y = -half;
    quat.rotateOut(s.pose.q, s.local, s.center1);
    s.local.y = half;
    quat.rotateOut(s.pose.q, s.local, s.center2);
    spring(world, eid, physics, s);
    collide(physics, handle.id.index1 - 1, s);
    push(physics, s);
    clipVector(velocity, s.planes, s.count, velocity);
    vc[offset] = velocity.x;
    vc[offset + 1] = velocity.y;
    vc[offset + 2] = velocity.z;
    vc[offset + 3] = 0;
    character.velocity.markChanged(eid);
    const pc = body.position.column;
    pc[offset] = position.x;
    pc[offset + 1] = position.y;
    pc[offset + 2] = position.z;
    pc[offset + 3] = 0;
    body.position.markChanged(eid);
    // The kinematic body mirrors the virtual capsule's endpoint; integrating it again would move it twice.
    if (
        !Object.is(position.x, s.start.x) ||
        !Object.is(position.y, s.start.y) ||
        !Object.is(position.z, s.start.z)
    )
        handle.setTransform(position, s.pose.q);
    handle.setLinearVelocity(s.zeroVelocity);
}

type Scratch = ReturnType<typeof characterScratch.create>;

function spring(world: World, eid: number, physics: PhysicsWorld, s: Scratch): void {
    const character = world.storage(Character);
    const state = physics.state;
    const q = queryColumns(state);
    const own = physics.getBody(eid)!.id.index1 - 1;
    const position = s.pose.p,
        velocity = s.velocity,
        up = s.up;
    const radius = world.storage(Body).halfExtents.column[eid * 4 + 3];
    const dt = s.dt;
    const rest = f32(3 * radius);
    const rayLength = f32(rest + radius);
    vec3.addOut(position, s.center1, s.ray);
    const k = q.prepare(s.ray);
    q.headerU[6] = 0;
    q.headerU[7] = 1;
    q.headerU[9] = 0xfffffffd;
    q.headerU[19] = own + 1;
    vec3.scaleOut(-rayLength, up, s.translation);
    q.translation(s.translation);
    k.worldQuery(state.worldId, 3, 0);
    const hit = q.resultU[0] !== 0xffffffff;
    s.groundShape = hit ? q.resultU[0] : -1;
    let pogo = character.pogoVelocity.column[eid];
    if (hit) {
        s.point.x = f32(s.ray.x + q.resultF[6]);
        s.point.y = f32(s.ray.y + q.resultF[7]);
        s.point.z = f32(s.ray.z + q.resultF[8]);
        groundVelocity(physics, q.resultU[0], s);
    } else {
        s.groundVelocity.x = 0;
        s.groundVelocity.y = 0;
        s.groundVelocity.z = 0;
    }
    // Upward platform carry is not a jump: pogo suppression uses the hit body's frame.
    vec3.subOut(velocity, s.groundVelocity, s.vr);
    if (!hit || vec3.dot(s.vr, up) > 0) {
        character.groundState.set(eid, GroundState.InAir);
        character.groundNormal.set(eid, 0, 0, 0, 0);
        character.groundVelocity.set(eid, 0, 0, 0, 0);
        pogo = 0;
    } else {
        s.normal.x = q.resultF[9];
        s.normal.y = q.resultF[10];
        s.normal.z = q.resultF[11];
        s.point.x = f32(s.ray.x + q.resultF[6]);
        s.point.y = f32(s.ray.y + q.resultF[7]);
        s.point.z = f32(s.ray.z + q.resultF[8]);
        character.groundState.set(
            eid,
            vec3.dot(s.normal, up) < Math.cos(character.maxSlope.column[eid])
                ? GroundState.OnSteepGround
                : GroundState.OnGround,
        );
        const offset = eid * 4;
        const normals = character.groundNormal.column;
        normals[offset] = s.normal.x;
        normals[offset + 1] = s.normal.y;
        normals[offset + 2] = s.normal.z;
        normals[offset + 3] = 0;
        character.groundNormal.markChanged(eid);
        const gv = s.groundVelocity;
        const velocities = character.groundVelocity.column;
        velocities[offset] = gv.x;
        velocities[offset + 1] = gv.y;
        velocities[offset + 2] = gv.z;
        velocities[offset + 3] = 0;
        character.groundVelocity.markChanged(eid);
        const omega = f32(f32(2 * PI) * 4);
        const omegaH = f32(omega * dt);
        const length = f32(q.resultF[5] * rayLength);
        pogo = f32(
            f32(pogo - f32(f32(omega * omegaH) * f32(length - rest))) /
                f32(f32(1 + f32(f32(f32(2 * f32(0.7)) * omegaH))) + f32(omegaH * omegaH)),
        );
    }
    character.pogoVelocity.column[eid] = pogo;
    character.pogoVelocity.markChanged(eid);
    vec3.mulAddOut(position, dt, velocity, s.target);
    vec3.mulAddOut(s.target, f32(dt * pogo), up, s.target);
}

function groundVelocity(physics: PhysicsWorld, shape: number, s: Scratch): void {
    const state = physics.state;
    const body = shapeBodyId(state, state.shapes[shape].id);
    const sim = getBodySim(state, body);
    const ground = getBodyState(state, body);
    const gv = s.groundVelocity;
    gv.x = 0;
    gv.y = 0;
    gv.z = 0;
    if (ground === null) return;
    readSimCenter(state, sim, s.center);
    readStateAngularVelocity(state, ground, s.angular);
    readStateLinearVelocity(state, ground, s.linear);
    vec3.subOut(s.point, s.center, s.r);
    vec3.crossOut(s.angular, s.r, gv);
    vec3.addOut(s.linear, gv, gv);
}

function collide(physics: PhysicsWorld, own: number, s: Scratch): void {
    const state = physics.state;
    const q = queryColumns(state);
    const k = q.prepare(s.pose.p);
    const position = s.pose.p;
    let count = 0;
    s.passes = 0;
    s.solverIterations = 0;
    for (let pass = 0; pass < 5; ++pass) {
        q.prepare(position);
        q.headerU[6] = 0;
        q.headerU[7] = 1;
        q.headerU[19] = own + 1;
        q.mover(s.center1, s.center2, s.radius);
        k.worldQuery(state.worldId, 5, 0);
        count = q.resultU[0];
        for (let i = 0; i < count; ++i) {
            const n = 16 + 8 * i;
            const plane = s.planes[i];
            plane.plane.normal.x = q.resultF[n + 1];
            plane.plane.normal.y = q.resultF[n + 2];
            plane.plane.normal.z = q.resultF[n + 3];
            plane.plane.offset = q.resultF[n + 4];
            s.points[i].x = f32(position.x + q.resultF[n + 5]);
            s.points[i].y = f32(position.y + q.resultF[n + 6]);
            s.points[i].z = f32(position.z + q.resultF[n + 7]);
            s.shapes[i] = q.resultU[n];
        }
        vec3.subOut(s.target, position, s.delta);
        const solved = solvePlanes(s.delta, s.planes, count, s.solved);
        s.passes++;
        s.solverIterations += solved.iterationCount;
        q.prepare(position);
        q.headerU[6] = 0;
        q.headerU[7] = 1;
        q.headerU[9] = 0xfffffffd;
        q.headerU[19] = own + 1;
        q.translation(solved.delta);
        k.worldQuery(state.worldId, 6, 0);
        vec3.scaleOut(q.resultF[3], solved.delta, s.delta);
        vec3.addOut(position, s.delta, position);
        if (vec3.lengthSq(s.delta) < f32(f32(0.01) * f32(0.01))) break;
    }
    s.count = count;
}

function push(physics: PhysicsWorld, s: Scratch): void {
    const state = physics.state;
    const velocity = s.velocity;
    s.impulseCount = 0;
    for (let i = 0; i < s.count; ++i) {
        const pushed = shapeBodyId(state, state.shapes[s.shapes[i]].id);
        if (bodyType(state, bodyField(state, pushed, BodyField.id)) !== BodyType.Dynamic) continue;
        const sim = getBodySim(state, pushed);
        const b = getBodyState(state, pushed);
        vec3.scaleOut(-1, s.planes[i].plane.normal, s.normal);
        readSimCenter(state, sim, s.center);
        vec3.subOut(s.points[i], s.center, s.r);
        vec3.crossOut(s.r, s.normal, s.rn);
        const m = readSimInvInertiaWorld(state, sim, s.inertia);
        const rn = s.rn;
        s.mrn.x = f32(f32(f32(m.cx.x * rn.x) + f32(m.cy.x * rn.y)) + f32(m.cz.x * rn.z));
        s.mrn.y = f32(f32(f32(m.cx.y * rn.x) + f32(m.cy.y * rn.y)) + f32(m.cz.y * rn.z));
        s.mrn.z = f32(f32(f32(m.cx.z * rn.x) + f32(m.cy.z * rn.y)) + f32(m.cz.z * rn.z));
        const mass = f32(simInvMass(state, sim) + vec3.dot(rn, s.mrn));
        const normalMass = mass > 0 ? f32(1 / mass) : 0;
        s.vr.x = 0;
        s.vr.y = 0;
        s.vr.z = 0;
        if (b !== null) {
            readStateAngularVelocity(state, b, s.angular);
            readStateLinearVelocity(state, b, s.linear);
            vec3.crossOut(s.angular, s.r, s.vr);
            vec3.addOut(s.linear, s.vr, s.vr);
        }
        vec3.subOut(s.vr, velocity, s.vr);
        const vn = vec3.dot(s.vr, s.normal);
        vec3.scaleOut(Math.max(f32(-normalMass * vn), 0), s.normal, s.impulse);
        vec3.mulSubOut(velocity, 0, s.impulse, velocity);
        let index = 0;
        while (
            index < s.impulseCount &&
            s.impulseBodies[index] !== bodyField(state, pushed, BodyField.id)
        )
            index++;
        if (index === s.impulseCount) {
            s.impulseCount++;
            s.impulseBodies[index] = bodyField(state, pushed, BodyField.id);
            s.impulses[index].x = 0;
            s.impulses[index].y = 0;
            s.impulses[index].z = 0;
        }
        vec3.addOut(s.impulses[index], s.impulse, s.impulses[index]);
        bodyApplyLinearImpulse(state, pushed, s.impulse, s.points[i], true);
    }
}

const MoveCharactersSystem: System = {
    name: "characters",
    group: "fixed",
    after: [SyncPhysicsConstraintsSystem],
    before: [StepPhysicsSystem],
    update(world) {
        for (const eid of world.query(terms)) moveCharacter(world, eid);
    },
};

/** Optional velocity-driven capsule movement, without gravity or input policy. */
export const CharacterPlugin: Plugin = {
    name: "Character",
    dependencies: [StandardPhysicsPlugin],
    components: [Character],
    systems: [MoveCharactersSystem],
};
