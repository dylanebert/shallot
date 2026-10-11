import {
    Body,
    BodyType,
    createApp,
    Shape,
    ShapeKind,
    StandardPhysicsPlugin,
} from "@dylanebert/shallot";
import {
    castRay,
    castRayClosest,
    castShape,
    castShapeClosest,
    overlapAABB,
    overlapShape,
    type PhysicsQueryHit,
} from "@dylanebert/shallot/standard/physics";
import { xf } from "../common/math";
import { defaultQueryFilter, defaultSurfaceMaterial } from "../common/types";
import { queryColumns } from "../kernel/querycolumns";
import { physicsState } from "../runtime";
import { createCompound } from "../shapes/compound";
import { createGrid } from "../shapes/heightfield";
import { makeBoxHull } from "../shapes/hull";
import { createGridMesh } from "../shapes/mesh";
import {
    createCapsuleShape,
    createCompoundShape,
    createHeightFieldShape,
    createHullShape,
    createMeshShape,
    createSphereShape,
} from "../shapes/shape";
import { createTestBody, testShapeDef } from "../testing";

export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};

export default async function create() {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const state = physicsState(app.world)!;
    const entityApp = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const entityWorld = entityApp.world;
    const hull = makeBoxHull(1, 0.25, 1);
    const material = defaultSurfaceMaterial();
    const compound = createCompound({ hulls: [{ hull, transform: xf.identity(), material }] })!;
    const capsule = {
        center1: { x: 0, y: -0.5, z: 0 },
        center2: { x: 0, y: 0.5, z: 0 },
        radius: 0.4,
    };
    const first = createTestBody(state, { position: { x: 0, y: 0, z: 0 } });
    createSphereShape(state, first, testShapeDef(), { center: { x: 0, y: 0, z: 0 }, radius: 1 });
    const second = createTestBody(state, { position: { x: 8, y: 0, z: 0 } });
    createCapsuleShape(state, second, testShapeDef(), capsule);
    const third = createTestBody(state, { position: { x: 16, y: 0, z: 0 } });
    createHullShape(state, third, testShapeDef(), hull);
    const fourth = createTestBody(state, { position: { x: 24, y: 0, z: 0 } });
    createMeshShape(state, fourth, testShapeDef(), createGridMesh(4, 4, 1, 0, true), {
        x: 1,
        y: 1,
        z: 1,
    });
    const fifth = createTestBody(state, { position: { x: 32, y: 0, z: 0 } });
    createHeightFieldShape(
        state,
        fifth,
        testShapeDef(),
        createGrid(8, 8, { x: 1, y: 1, z: 1 }, false),
    );
    const sixth = createTestBody(state, { position: { x: 40, y: 0, z: 0 } });
    createCompoundShape(state, sixth, testShapeDef(), compound);
    const origin = { x: 0, y: 0.75, z: 0 };
    const rayOrigin = { x: 0, y: 3, z: 0 };
    const translation = { x: 0, y: -4, z: 0 };
    const filter = defaultQueryFilter();
    const q = queryColumns(state);
    const body = entityWorld.create();
    entityWorld.add(body, Body, { type: BodyType.Static });
    const shape = entityWorld.create();
    entityWorld.add(shape, Shape, {
        body,
        kind: ShapeKind.Sphere,
        sphere: [0, 0, 0, 1],
    });
    entityWorld.step(1 / 60);
    const queryOrigin = { x: -3, y: 0, z: 0 };
    const overlapOrigin = { x: 0, y: 0, z: 0 };
    const queryTranslation = { x: 6, y: 0, z: 0 };
    const queryBounds = {
        lowerBound: { x: -1, y: -1, z: -1 },
        upperBound: { x: 1, y: 1, z: 1 },
    };
    const proxy = { points: [{ x: 0, y: 0, z: 0 }], count: 1, radius: 0.25 };
    const output: PhysicsQueryHit = {
        shape: 0,
        body: 0,
        point: { x: 0, y: 0, z: 0 },
        normal: { x: 0, y: 0, z: 0 },
        fraction: 0,
        userMaterialId: 0n,
        triangleIndex: 0,
        childIndex: 0,
    };
    const cast = (hit: PhysicsQueryHit) => {
        if (hit.shape !== shape || hit.body !== body)
            throw new Error("ECS ray query lost its Shape or Body entity");
        return hit.fraction;
    };
    const overlap = (shapeEid: number, bodyEid: number) => {
        if (shapeEid !== shape || bodyEid !== body)
            throw new Error("ECS overlap query lost its Shape or Body entity");
        return true;
    };
    return {
        step: () => {
            const k = q.prepare(origin, filter);
            q.mover(capsule.center1, capsule.center2, capsule.radius);
            q.translation(translation);
            for (let i = 0; i < 6; ++i) {
                q.headerF[10] = 8 * i;
                q.headerF[11] = origin.y;
                k.worldQuery(state.worldId, 5, 0);
                q.headerF[11] = rayOrigin.y;
                k.worldQuery(state.worldId, 6, 0);
                k.worldQuery(state.worldId, 3, 0);
            }
            castRay(entityWorld, queryOrigin, queryTranslation, cast, filter);
            castRayClosest(entityWorld, queryOrigin, queryTranslation, output, filter);
            castShape(entityWorld, queryOrigin, proxy, queryTranslation, cast, filter);
            castShapeClosest(entityWorld, queryOrigin, proxy, queryTranslation, output, filter);
            overlapAABB(entityWorld, queryBounds, overlap, filter);
            overlapShape(entityWorld, overlapOrigin, proxy, overlap, filter);
            entityWorld.step(1 / 60);
        },
        dispose: async () => {
            await app.dispose();
            await entityApp.dispose();
        },
    };
}
