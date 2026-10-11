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
import { physicsWorld } from "../runtime";
import { createCompound } from "../shapes/compound";
import { createGrid } from "../shapes/heightfield";
import { makeBoxHull } from "../shapes/hull";
import { createGridMesh } from "../shapes/mesh";

export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};

export default async function create() {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const physics = physicsWorld(app.world)!;
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
    physics
        .createBody({ position: { x: 0, y: 0, z: 0 } })
        .createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 1 });
    physics.createBody({ position: { x: 8, y: 0, z: 0 } }).createCapsule({}, capsule);
    physics.createBody({ position: { x: 16, y: 0, z: 0 } }).createHull({}, hull);
    physics
        .createBody({ position: { x: 24, y: 0, z: 0 } })
        .createMesh({}, createGridMesh(4, 4, 1, 0, true));
    physics
        .createBody({ position: { x: 32, y: 0, z: 0 } })
        .createHeightField({}, createGrid(8, 8, { x: 1, y: 1, z: 1 }, false));
    physics.createBody({ position: { x: 40, y: 0, z: 0 } }).createCompound({}, compound);
    const origin = { x: 0, y: 0.75, z: 0 };
    const rayOrigin = { x: 0, y: 3, z: 0 };
    const translation = { x: 0, y: -4, z: 0 };
    const filter = defaultQueryFilter();
    const q = queryColumns(physics.state);
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
                k.worldQuery(physics.state.worldId, 5, 0);
                q.headerF[11] = rayOrigin.y;
                k.worldQuery(physics.state.worldId, 6, 0);
                k.worldQuery(physics.state.worldId, 3, 0);
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
