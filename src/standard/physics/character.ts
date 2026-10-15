import { Body, Shape as ShapeComponent, ShapeKind } from "../../core/physics";
import {
    component,
    f32 as field,
    type Plugin,
    type System,
    Time,
    u32,
    type World,
} from "../../engine";
import * as ecs from "../../engine/ecs";
import { vec3, xf } from "./common/math";
import { BodyType } from "./common/types";
import { BodyField, setBodyField } from "./kernel/bodyrecords";
import { queryColumns } from "./kernel/querycolumns";
import {
    physicsBodyIndex,
    physicsState,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    SyncPhysicsConstraintsSystem,
} from "./runtime";
import {
    bodySetAwake,
    bodySetLinearVelocity,
    bodySetTransform,
    readBodyTransform,
} from "./world/body";

export const GroundState = { InAir: 0, OnGround: 1, OnSteepGround: 2 } as const;
/** A kinematic capsule Body. Write velocity in a fixed system ordered before `CharacterPlugin.systems`; movement runs after body synchronization and before the rigid solver. Resolved velocity is written back, without gravity or input policy. Ground velocity is reported, never added to motion. */
export const Character = component(
    "Character",
    {
        /** World-space metres per second. */
        velocity: ecs.vec3,
        groundState: u32,
        /** World-space unit normal at the pogo ray hit, or zero in air. */
        groundNormal: ecs.vec3,
        /** Hit body's velocity at the ground point, in world-space metres per second, or zero in air. */
        groundVelocity: ecs.vec3,
        /** Maximum walkable slope in radians, measured from up. */
        maxSlope: field,
        /** Unit world-space up direction. */
        up: ecs.vec3,
        /** Pogo spring speed in metres per second; persists between fixed ticks. */
        pogoVelocity: field,
        /** Pogo spring rest length in capsule radii, down from the lower sphere's centre, so the capsule rests `pogoRest - 1` radii above contact. Box3D's mover sample uses 3. */
        pogoRest: field,
        /** 1 adds a vertical wall behind each contact steeper than `maxSlope`, so a rounded capsule cannot ride up a step edge, as Jolt's `CharacterVirtual` does. Box3D's mover sample does not, so 0 is the default. */
        steepWalls: u32,
    },
    {
        requires: [Body],
        defaults: () => ({
            velocity: [0, 0, 0],
            groundState: GroundState.InAir,
            groundNormal: [0, 0, 0],
            groundVelocity: [0, 0, 0],
            maxSlope: Math.PI / 4,
            up: [0, 1, 0],
            pogoVelocity: 0,
            pogoRest: 3,
            steepWalls: 0,
        }),
    },
);
const terms = [Character, Body];

export const characterScratch = {
    create: () => ({
        pose: xf.identity(),
        start: vec3.zero(),
        zeroVelocity: vec3.zero(),
        center: vec3.zero(),
        input: new Float32Array(0),
        inputU: new Uint32Array(0),
        slope: new Float64Array(0),
        output: new Float32Array(0),
        outputU: new Uint32Array(0),
        shapes: new Int32Array(8),
        impulseBodies: new Int32Array(8),
        impulses: Array.from({ length: 8 }, () => vec3.zero()),
        impulseCount: 0,
        count: 0,
        planeCount: 0,
        groundShape: -1,
        passes: 0,
        solverIterations: 0,
    }),
};

function moveCharacter(world: World, eid: number): void {
    const state = physicsState(world);
    const bodyId = physicsBodyIndex(world, eid);
    if (!state || bodyId < 0) return;
    const body = world.storage(Body);
    const shape = world.storage(ShapeComponent);
    if (
        !world.has(eid, ShapeComponent) ||
        shape.kind.column[eid] !== ShapeKind.Capsule ||
        body.type.column[eid] !== BodyType.Kinematic
    )
        throw new Error(`Character entity ${eid} requires a kinematic capsule Body`);
    const character = world.storage(Character);
    const s = world.resource(characterScratch);
    // Pose-driven kinematics must stay in the solver's live publish set, without periodic wake/clone churn.
    bodySetAwake(state, bodyId, true);
    setBodyField(state, bodyId, BodyField.sleepTime, 0);
    readBodyTransform(state, bodyId, s.pose);
    const position = s.pose.p;
    s.start.x = position.x;
    s.start.y = position.y;
    s.start.z = position.z;
    const k = queryColumns(state).prepare(position);
    const memory = k.memory.buffer;
    if (s.input.buffer !== memory) {
        s.input = new Float32Array(memory, k.characterInputPtr(), 30);
        s.inputU = new Uint32Array(memory, k.characterInputPtr(), 30);
        s.slope = new Float64Array(memory, k.characterInputPtr() + 80, 1);
        s.output = new Float32Array(memory, k.characterOutputPtr(), 63);
        s.outputU = new Uint32Array(memory, k.characterOutputPtr(), 63);
    }
    const input = s.input;
    const offset = eid * 4;
    const velocity = character.velocity.column;
    const up = character.up.column;
    const a = shape.capsuleA.column;
    const b = shape.capsuleB.column;
    input[0] = position.x;
    input[1] = position.y;
    input[2] = position.z;
    for (let i = 0; i < 3; ++i) {
        input[3 + i] = velocity[offset + i];
        input[6 + i] = up[offset + i];
        input[9 + i] = a[offset + i];
        input[12 + i] = b[offset + i];
    }
    input[15] = b[offset + 3];
    input[16] = Time.FIXED_DT;
    input[17] = character.pogoVelocity.column[eid];
    input[18] = character.pogoRest.column[eid];
    s.inputU[19] = character.steepWalls.column[eid];
    s.slope[0] = character.maxSlope.column[eid];
    input[22] = s.center.x;
    input[23] = s.center.y;
    input[24] = s.center.z;
    input[25] = s.pose.q.v.x;
    input[26] = s.pose.q.v.y;
    input[27] = s.pose.q.v.z;
    input[28] = s.pose.q.s;
    input[29] = state.maxLinearSpeed;
    k.characterMove(state.worldId, bodyId);
    if (s.output.buffer !== k.memory.buffer) {
        s.output = new Float32Array(k.memory.buffer, k.characterOutputPtr(), 63);
        s.outputU = new Uint32Array(k.memory.buffer, k.characterOutputPtr(), 63);
    }
    const result = s.output,
        words = s.outputU;
    position.x = result[0];
    position.y = result[1];
    position.z = result[2];
    const normals = character.groundNormal.column;
    const groundVelocity = character.groundVelocity.column;
    const positions = body.position.column;
    for (let i = 0; i < 3; ++i) {
        velocity[offset + i] = result[3 + i];
        normals[offset + i] = result[8 + i];
        groundVelocity[offset + i] = result[11 + i];
        positions[offset + i] = result[i];
    }
    velocity[offset + 3] = 0;
    normals[offset + 3] = 0;
    groundVelocity[offset + 3] = 0;
    positions[offset + 3] = 0;
    character.velocity.markChanged(eid);
    character.groundNormal.markChanged(eid);
    character.groundVelocity.markChanged(eid);
    body.position.markChanged(eid);
    character.pogoVelocity.column[eid] = result[6];
    character.pogoVelocity.markChanged(eid);
    character.groundState.set(eid, words[7]);
    s.count = words[14];
    s.planeCount = words[15];
    s.passes = words[16];
    s.solverIterations = words[17];
    s.impulseCount = words[18];
    for (let i = 0; i < 8; ++i) {
        s.impulseBodies[i] = words[19 + i];
        s.impulses[i].x = result[27 + 3 * i];
        s.impulses[i].y = result[28 + 3 * i];
        s.impulses[i].z = result[29 + 3 * i];
        s.shapes[i] = words[55 + i];
    }
    s.center.x = result[51];
    s.center.y = result[52];
    s.center.z = result[53];
    s.groundShape = words[54] === 0xffffffff ? -1 : words[54];
    // The kinematic body mirrors the virtual capsule's endpoint; integrating it again would move it twice.
    if (
        !Object.is(position.x, s.start.x) ||
        !Object.is(position.y, s.start.y) ||
        !Object.is(position.z, s.start.z)
    )
        bodySetTransform(state, bodyId, position, s.pose.q);
    bodySetLinearVelocity(state, bodyId, s.zeroVelocity);
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
    // Query and motion scratch is overwritten per call; Character fields and physics own recovery.
    recovery: "stateless",
    dependencies: [StandardPhysicsPlugin],
    components: [Character],
    systems: [MoveCharactersSystem],
};
