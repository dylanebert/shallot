import type { PhysicsWorld } from "./api/world";
import { updateBroadPhasePairs } from "./collision/pairs";
import { BodyType, defaultShapeDef } from "./common/types";
import { createCompound } from "./shapes/compound";
import { createGrid } from "./shapes/heightfield";
import { makeBoxHull } from "./shapes/hull";
import { createGridMesh } from "./shapes/mesh";
import {
    createCapsuleShape,
    createHeightFieldShape,
    createHullShape,
    createMeshShape,
    createSphereShape,
    destroyShape,
    setShapeFilter,
} from "./shapes/shape";
import { bodyDisable, bodyEnable } from "./world/body";

export function shapeAllocationSubject(world: PhysicsWorld, rebuildTopology = true): () => void {
    const body = world.createBody({ type: BodyType.Dynamic, position: { x: 10, y: 0, z: 0 } });
    const id = body.id.index1 - 1;
    const staticBody = world.createBody({ type: BodyType.Static, position: { x: 30, y: 0, z: 0 } });
    const staticId = staticBody.id.index1 - 1;
    const state = world.state;
    const def = defaultShapeDef();
    def.density = 2.5;
    const staticDef = defaultShapeDef();
    const sphere = { center: { x: 0.2, y: 0.3, z: 0.4 }, radius: 0.5 };
    const capsule = {
        center1: { x: 0, y: -0.3, z: 0 },
        center2: { x: 0.1, y: 0.4, z: 0.2 },
        radius: 0.3,
    };
    const hull = makeBoxHull(0.5, 0.75, 0.25);
    const mesh = createGridMesh(2, 2, 1, 1, true);
    const scale = { x: -1.25, y: 1, z: 0.75 };
    const heightField = createGrid(3, 3, { x: 1, y: 1, z: 1 }, false);
    const compound = createCompound({ spheres: [{ sphere, material: staticDef.baseMaterial }] })!;
    // Keep the owned geometry resident, so recycling records does not grow a geometry database.
    staticBody.createSphere({}, sphere);
    staticBody.createHull({}, hull);
    staticBody.createMesh({}, mesh, scale);
    staticBody.createHeightField({}, heightField);
    staticBody.createCompound({}, compound);
    world.step(1 / 60, 1);
    const filterA = {
        categoryBits: 0x8000000100000001n,
        maskBits: 0xffffffffffffffffn,
        groupIndex: 1,
    };
    const filterB = {
        categoryBits: 0xffffffffffffffffn,
        maskBits: 0x8000000100000001n,
        groupIndex: 0,
    };
    const negativeMaterialId = -1n;
    const wideMaterialId = 0x10000000000000001n;
    let serial = 0;
    return () => {
        for (let i = 0; i < 16; ++i) {
            const kind = serial++ % 3;
            def.isSensor = (i & 3) === 0;
            const shape =
                kind === 0
                    ? createSphereShape(state, id, def, sphere)!
                    : kind === 1
                      ? createCapsuleShape(state, id, def, capsule)!
                      : createHullShape(state, id, def, hull)!;
            const other =
                kind === 0
                    ? createMeshShape(state, staticId, staticDef, mesh, scale)!
                    : createHeightFieldShape(state, staticId, staticDef, heightField)!;
            setShapeFilter(state, shape, filterA);
            setShapeFilter(state, shape, filterB);
            def.baseMaterial.userMaterialId = (i & 1) === 0 ? negativeMaterialId : wideMaterialId;
            state.shapeStore.writeMaterials(state, shape, def.baseMaterial);
            bodyDisable(state, staticId);
            bodyEnable(state, staticId);
            if (rebuildTopology) {
                bodyDisable(state, id);
                bodyEnable(state, id);
                body.setAwake(false);
                body.setAwake(true);
            }
            updateBroadPhasePairs(state);
            destroyShape(state, shape, true);
            destroyShape(state, other, true);
        }
    };
}
