import { expect, test } from "bun:test";
import { Body, PhysicsPlugin, ShapeKind } from "../../core/physics";
import { Time, World } from "../../engine";
import {
    BodyType,
    createHeightField,
    createHull,
    createMesh,
    type HullData,
    type MeshData,
    makeBoxHull,
    type Body as SolverBody,
} from "./api";
import { characterScratch } from "./character";
import {
    Character,
    CharacterPlugin,
    GroundState,
    PhysicsWorld,
    PhysicsWorldDefinition,
    physicsWorld,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
} from "./index";
import lane from "./oracle/box3d/47d7f7cc7e091142c08d11dc7d2e493c5d34f536/v7/cases.json";

type Command = { op: string; id: string; [key: string]: unknown };
const word = new Uint32Array(1);
const scalar = new Float32Array(word.buffer);
function f32(value: unknown): number {
    word[0] = Number.parseInt(String(value).slice(2), 16);
    return scalar[0];
}
function bits(value: number): string {
    scalar[0] = value;
    return `0x${word[0].toString(16).padStart(8, "0")}`;
}
function vector(value: unknown) {
    const values = value as string[];
    return { x: f32(values[0]), y: f32(values[1]), z: f32(values[2]) };
}
const vectorBits = (value: { x: number; y: number; z: number }) => [
    bits(value.x),
    bits(value.y),
    bits(value.z),
];

// The only exclusion is the soft-plane variant (MoverShapeUserData), outside Character's Scope.
const excluded = lane.cases.filter((item) =>
    item.input.commands.some((command) => command.op === "mover.solve" && !command.clipVelocity),
);
test("the mover lane excludes exactly the out-of-scope soft-plane case", () => {
    expect(excluded.map((item) => item.id)).toEqual(["m1.wall-unclipped.v1"]);
});
for (const item of lane.cases) {
    if (excluded.includes(item)) continue;
    test(`CharacterPlugin move matches ${item.id} bit-for-bit`, async () => {
        const world = new World();
        await StandardPhysicsPlugin.initialize!(world);
        await StandardPhysicsPlugin.warm!(world);
        for (const plugin of [PhysicsPlugin, CharacterPlugin])
            for (const component of plugin.components!) world.registry.register(component);
        for (const system of [...StandardPhysicsPlugin.systems!, ...CharacterPlugin.systems!])
            world.addSystem(system);
        const fixedDt = Time.FIXED_DT;
        const physics = physicsWorld(world)!;
        const bodies = new Map<string, SolverBody>();
        const hulls = new Map<string, HullData>();
        const meshes = new Map<string, MeshData>();
        const fields = new Map<string, ReturnType<typeof createHeightField>>();
        const spheres = new Map<
            string,
            { center: { x: number; y: number; z: number }; radius: number }
        >();
        try {
            for (const command of item.input.commands as Command[]) {
                switch (command.op) {
                    case "world.create":
                        world.resource(PhysicsWorldDefinition).gravity = vector(command.gravity);
                        break;
                    case "body.create":
                        bodies.set(
                            command.id,
                            physics.createBody({
                                type:
                                    command.type === "dynamic" ? BodyType.Dynamic : BodyType.Static,
                                position: vector(command.position),
                                linearVelocity: vector(command.linearVelocity),
                                angularVelocity: vector(command.angularVelocity),
                            }),
                        );
                        break;
                    case "resource.box": {
                        const half = vector(command.halfExtents);
                        hulls.set(command.id, makeBoxHull(half.x, half.y, half.z));
                        break;
                    }
                    case "resource.hull": {
                        const points = (command.points as unknown[]).map(vector);
                        const hull = createHull(points, points.length);
                        if (!hull) throw new Error(`invalid hull ${command.id}`);
                        hulls.set(command.id, hull);
                        break;
                    }
                    case "resource.mesh": {
                        const values = (command.vertices as unknown[]).map(f32);
                        const vertices = [];
                        for (let i = 0; i < values.length; i += 3)
                            vertices.push({ x: values[i], y: values[i + 1], z: values[i + 2] });
                        const mesh = createMesh({
                            vertices,
                            indices: command.indices as number[],
                            identifyEdges: command.identifyEdges === true,
                        });
                        if (!mesh) throw new Error(`invalid mesh ${command.id}`);
                        meshes.set(command.id, mesh);
                        break;
                    }
                    case "resource.height-field":
                        fields.set(
                            command.id,
                            createHeightField({
                                heights: (command.samples as unknown[]).map(f32),
                                materialIndices: null,
                                clockwiseWinding: command.clockwiseWinding === true,
                                scale: {
                                    x: f32(command.scaleX),
                                    y: f32(command.scaleY),
                                    z: f32(command.scaleZ),
                                },
                                countX: Number(command.countX),
                                countZ: Number(command.countZ),
                                globalMinimumHeight: f32(command.globalMinimumHeight),
                                globalMaximumHeight: f32(command.globalMaximumHeight),
                            }),
                        );
                        break;
                    case "resource.sphere":
                        spheres.set(command.id, {
                            center: { x: 0, y: 0, z: 0 },
                            radius: f32(command.radius),
                        });
                        break;
                    case "shape.create": {
                        const body = bodies.get(String(command.body))!;
                        const resource = String(command.resource);
                        const def =
                            command.density === undefined ? {} : { density: f32(command.density) };
                        if (command.kind === "mesh") body.createMesh(def, meshes.get(resource)!);
                        else if (command.kind === "height-field")
                            body.createHeightField(def, fields.get(resource)!);
                        else if (command.kind === "sphere")
                            body.createSphere(def, spheres.get(resource)!);
                        else body.createHull(def, hulls.get(resource)!);
                        break;
                    }
                    case "mover.solve": {
                        const position = vector(command.position);
                        const velocity = vector(command.velocity);
                        const center1 = vector(command.center1);
                        const center2 = vector(command.center2);
                        expect(center1).toEqual({ x: 0, y: -0.5, z: 0 });
                        expect(center2).toEqual({ x: 0, y: 0.5, z: 0 });
                        const eid = world.create();
                        world.add(eid, Body, {
                            type: BodyType.Kinematic,
                            shape: ShapeKind.Capsule,
                            position: [position.x, position.y, position.z, 0],
                            halfExtents: [0, 0.5, 0, f32(command.radius)],
                        });
                        world.add(eid, Character, {
                            velocity: [velocity.x, velocity.y, velocity.z, 0],
                            pogoVelocity: f32(command.pogoVelocity),
                        });
                        // The oracle records SolveMove before the rigid-body step, with its own fixed duration.
                        Object.defineProperty(Time, "FIXED_DT", { value: f32(command.timeStep) });
                        let observed = false;
                        let failure: unknown;
                        world.addSystem({
                            name: "observe-mover",
                            group: "fixed",
                            after: CharacterPlugin.systems!,
                            before: [StepPhysicsSystem],
                            update() {
                                observed = true;
                                try {
                                    const expected = item.output.observations[0];
                                    const scratch = world.resource(characterScratch);
                                    expect(bits(scratch.count)).toBe(expected.planeCount);
                                    expect(scratch.passes).toBe(expected.passes);
                                    expect(scratch.solverIterations).toBe(
                                        expected.solverIterations,
                                    );
                                    expect(scratch.impulseCount).toBe(expected.impulses.length);
                                    const pose = world.storage(Body).position;
                                    const character = world.storage(Character);
                                    expect([
                                        bits(pose.x.get(eid)),
                                        bits(pose.y.get(eid)),
                                        bits(pose.z.get(eid)),
                                    ]).toEqual(expected.position);
                                    expect([
                                        bits(character.velocity.x.get(eid)),
                                        bits(character.velocity.y.get(eid)),
                                        bits(character.velocity.z.get(eid)),
                                    ]).toEqual(expected.velocity);
                                    expect(bits(character.pogoVelocity.get(eid))).toBe(
                                        expected.pogoVelocity,
                                    );
                                    expect(
                                        bits(
                                            character.groundState.get(eid) === GroundState.InAir
                                                ? 0
                                                : 1,
                                        ),
                                    ).toBe(expected.onGround);
                                    for (const impulse of expected.impulses) {
                                        const pushed = bodies.get(impulse.body)!;
                                        const index = Array.from(
                                            scratch.impulseBodies.subarray(0, scratch.impulseCount),
                                        ).indexOf(pushed.id.index1 - 1);
                                        expect(index).toBeGreaterThanOrEqual(0);
                                        expect(vectorBits(scratch.impulses[index])).toEqual(
                                            impulse.impulse,
                                        );
                                        const initial = (item.input.commands as Command[]).find(
                                            (c) => c.op === "body.create" && c.id === impulse.body,
                                        )!;
                                        const shape = (item.input.commands as Command[]).find(
                                            (c) =>
                                                c.op === "shape.create" && c.body === impulse.body,
                                        )!;
                                        const comparison = new PhysicsWorld({
                                            gravity: { x: 0, y: 0, z: 0 },
                                        });
                                        try {
                                            const twin = comparison.createBody({
                                                type: BodyType.Dynamic,
                                                position: vector(initial.position),
                                                linearVelocity: vector(initial.linearVelocity),
                                                angularVelocity: vector(initial.angularVelocity),
                                            });
                                            twin.createSphere(
                                                { density: f32(shape.density) },
                                                spheres.get(String(shape.resource))!,
                                            );
                                            // The point does not enter linear velocity; this lane's sphere impulse is radial.
                                            twin.applyLinearImpulse(
                                                vector(impulse.impulse),
                                                vector(initial.position),
                                                true,
                                            );
                                            expect(vectorBits(pushed.getLinearVelocity())).toEqual(
                                                vectorBits(twin.getLinearVelocity()),
                                            );
                                            expect(vectorBits(pushed.getAngularVelocity())).toEqual(
                                                vectorBits(twin.getAngularVelocity()),
                                            );
                                        } finally {
                                            comparison.destroy();
                                        }
                                    }
                                } catch (error) {
                                    failure = error;
                                }
                            },
                        });
                        world.step(Time.FIXED_DT);
                        // World steps report system errors rather than throwing them to the test runner.
                        if (failure !== undefined) throw failure;
                        expect(observed).toBe(true);
                        const finalPose = world.storage(Body).position;
                        expect([
                            bits(finalPose.x.get(eid)),
                            bits(finalPose.y.get(eid)),
                            bits(finalPose.z.get(eid)),
                        ]).toEqual(item.output.observations[0].position);
                        expect(vectorBits(physics.getBody(eid)!.getPosition())).toEqual(
                            item.output.observations[0].position,
                        );
                        break;
                    }
                    default:
                        throw new Error(`unreplayed command ${command.op}`);
                }
            }
        } finally {
            Object.defineProperty(Time, "FIXED_DT", { value: fixedDt });
            await StandardPhysicsPlugin.dispose!(world);
            world.dispose();
        }
    });
}
