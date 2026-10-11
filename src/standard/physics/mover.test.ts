import { expect, test } from "bun:test";
import { World } from "@dylanebert/shallot";
import { Body, PhysicsPlugin, Shape } from "@dylanebert/shallot/physics";
import {
    type Capsule,
    type CollisionPlane,
    clipVector,
    type PlaneResult,
    type PlaneSolverResult,
    type QueryFilter,
    StandardPhysicsPlugin,
    solvePlanes,
} from "@dylanebert/shallot/standard/physics";
import { rethrowQueryError, setQueryCallback } from "./kernel/kernel";
import { queryColumns } from "./kernel/querycolumns";
import { physicsState } from "./runtime";

function collideMover(
    world: World,
    origin: { x: number; y: number; z: number },
    mover: Capsule,
    callback: (planes: PlaneResult[]) => boolean,
    filter: QueryFilter,
) {
    const state = physicsState(world)!;
    const q = queryColumns(state);
    const k = q.prepare(origin, filter);
    q.mover(mover.center1, mover.center2, mover.radius);
    const previous = setQueryCallback(state.ecsState, (_kind, _id, data, count) => {
        state.shapeStore.refreshViews();
        const f = state.shapeStore.materialF;
        const base = data >>> 2;
        const planes: PlaneResult[] = [];
        for (let i = 0; i < count; ++i) {
            const n = base + i * 10;
            planes.push({
                plane: { normal: { x: f[n]!, y: f[n + 1]!, z: f[n + 2]! }, offset: f[n + 3]! },
                point: { x: f[n + 4]!, y: f[n + 5]!, z: f[n + 6]! },
                triangleIndex: f[n + 7]!,
                childIndex: f[n + 8]!,
                materialIndex: f[n + 9]!,
            });
        }
        return Number(callback(planes));
    });
    try {
        k.worldQuery(state.worldId, 5, 1);
        rethrowQueryError(state.ecsState);
    } finally {
        setQueryCallback(state.ecsState, previous);
    }
}

function castMover(
    world: World,
    origin: { x: number; y: number; z: number },
    mover: Capsule,
    translation: { x: number; y: number; z: number },
    filter: QueryFilter,
    accept: () => boolean,
): number {
    const state = physicsState(world)!;
    const q = queryColumns(state);
    const k = q.prepare(origin, filter);
    q.mover(mover.center1, mover.center2, mover.radius);
    q.translation(translation);
    const previous = setQueryCallback(state.ecsState, () => Number(accept()));
    try {
        k.worldQuery(state.worldId, 6, 1);
        rethrowQueryError(state.ecsState);
    } finally {
        setQueryCallback(state.ecsState, previous);
    }
    return q.resultF[3]!;
}

test("internal mover queries land a falling capsule on an authored floor sliding along a wall", async () => {
    const world = new World();
    for (const component of PhysicsPlugin.components!)
        world.registry.register(component, PhysicsPlugin.name);
    await StandardPhysicsPlugin.initialize!(world);
    await StandardPhysicsPlugin.warm!(world);
    for (const system of StandardPhysicsPlugin.systems!) world.addSystem(system);
    try {
        const floor = world.create();
        world.add(floor, Body, { position: [0, -0.5, 0, 0], isEnabled: 1 });
        world.add(floor, Shape, { scale: [10, 0.5, 10, 0] });
        const wall = world.create();
        world.add(wall, Body, { position: [1.5, 2, 0, 0], isEnabled: 1 });
        world.add(wall, Shape, { scale: [0.5, 2, 10, 0] });
        world.step(1 / 60);
        const capsule: Capsule = {
            center1: { x: 0, y: -0.5, z: 0 },
            center2: { x: 0, y: 0.5, z: 0 },
            radius: 0.3,
        };
        const filter: QueryFilter = { categoryBits: 1n, maskBits: 0xffffffffffffffffn };
        const accept = () => true;
        let planes: CollisionPlane[] = [];
        const gather = (results: PlaneResult[]) => {
            for (const result of results) {
                planes.push({
                    plane: result.plane,
                    pushLimit: 3.4028234663852886e38,
                    push: 0,
                    clipVelocity: true,
                });
            }
            return true;
        };
        let position = { x: 0, y: 3, z: -2 };
        let velocity = { x: 2, y: -4, z: 1 };
        let touchedFloor = false;
        let touchedWall = false;
        for (let tick = 0; tick < 90; ++tick) {
            velocity.y -= 10 / 60;
            const target = {
                x: position.x + velocity.x / 60,
                y: position.y + velocity.y / 60,
                z: position.z + velocity.z / 60,
            };
            // Box3D CharacterMover's collide, solve and cast passes, without its pogo spring or input tuning.
            for (let pass = 0; pass < 5; ++pass) {
                planes = [];
                collideMover(world, position, capsule, gather, filter);
                touchedFloor ||= planes.some(({ plane }) => plane.normal.y > 0.9);
                touchedWall ||= planes.some(({ plane }) => plane.normal.x < -0.9);
                const result: PlaneSolverResult = solvePlanes(
                    {
                        x: target.x - position.x,
                        y: target.y - position.y,
                        z: target.z - position.z,
                    },
                    planes,
                    planes.length,
                );
                const fraction = castMover(world, position, capsule, result.delta, filter, accept);
                const delta = {
                    x: fraction * result.delta.x,
                    y: fraction * result.delta.y,
                    z: fraction * result.delta.z,
                };
                position = {
                    x: position.x + delta.x,
                    y: position.y + delta.y,
                    z: position.z + delta.z,
                };
                if (delta.x * delta.x + delta.y * delta.y + delta.z * delta.z < 0.01 * 0.01) break;
            }
            velocity = clipVector(velocity, planes, planes.length);
        }
        expect(touchedFloor).toBe(true);
        expect(touchedWall).toBe(true);
        expect(position.x).toBeCloseTo(0.7, 2);
        // The plane solver permits Box3D's 0.005 m linear slop at the floor.
        expect(position.y).toBeCloseTo(0.795, 5);
        expect(position.z).toBeCloseTo(-0.5, 4);
        expect(velocity.x).toBeCloseTo(0, 5);
        expect(velocity.y).toBeCloseTo(0, 5);
        expect(velocity.z).toBe(1);
    } finally {
        await StandardPhysicsPlugin.dispose!(world);
        world.dispose();
    }
});
