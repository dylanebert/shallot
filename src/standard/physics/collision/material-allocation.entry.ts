import { createApp, StandardPhysicsPlugin } from "@dylanebert/shallot";
import type { Vec3 } from "../common/records";
import { BodyType, defaultSurfaceMaterial } from "../common/types";
import { kernel } from "../kernel/kernel";
import { physicsState } from "../runtime";
import { makeBoxHull } from "../shapes/build-hull";
import { createCompound } from "../shapes/compound";
import { createMesh } from "../shapes/mesh";
import { createCompoundShape, createHullShape, createMeshShape } from "../shapes/shape";
import { createTestBody, testShapeDef } from "../testing";
import { bodySetLinearVelocity, bodySetTransform } from "../world/body";
import { PhysicsWorldDefinition } from "../world-definition";
import { awakeContactCount } from "./contact";

export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};

export default async function create(input: string) {
    const [kind, startText] = input.split(":");
    const start = startText === undefined ? 0 : Number(startText);
    const count = startText === undefined ? 64 : 16;
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const world = app.world;
    const state = physicsState(world)!;
    const definition = world.resource(PhysicsWorldDefinition);
    definition.enableSleep = false;
    definition.contactRecycleDistance = 0;
    const k = kernel(state.ecsState);
    let mixes = 0;
    definition.frictionCallback = (_world, a, _ia, b, _ib) => {
        mixes++;
        return k.materialMixDefault(0, a, b);
    };
    definition.restitutionCallback = (_world, a, _ia, b, _ib) => k.materialMixDefault(1, a, b);
    const material = { ...defaultSurfaceMaterial(), userMaterialId: 0x123456789abcdef0n };
    const ground = createTestBody(state, { type: BodyType.Static });
    if (kind === "mesh") {
        const mesh = createMesh({
            vertices: [
                { x: -100, y: 0, z: -100 },
                { x: 100, y: 0, z: -100 },
                { x: 100, y: 0, z: 100 },
                { x: -100, y: 0, z: 100 },
            ],
            indices: [0, 2, 1, 0, 3, 2],
            identifyEdges: true,
        });
        if (!mesh) throw new Error("mesh construction failed");
        createMeshShape(state, ground, testShapeDef({ baseMaterial: material }), mesh, {
            x: 1,
            y: 1,
            z: 1,
        });
    } else {
        const compound = createCompound({
            hulls: [
                {
                    hull: makeBoxHull(100, 0.5, 100),
                    material,
                    transform: { p: { x: 0, y: -0.5, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
                },
            ],
        });
        if (!compound) throw new Error("compound construction failed");
        createCompoundShape(state, ground, testShapeDef(), compound);
    }
    const hull = makeBoxHull(0.5, 0.5, 0.5);
    const boxes: { body: number; position: Vec3 }[] = [];
    for (let offset = 0; offset < count; offset++) {
        const i = start + offset;
        const position = { x: (i % 8) * 3 - 12, y: 0.5, z: Math.floor(i / 8) * 3 - 12 };
        const body = createTestBody(state, { type: BodyType.Dynamic, position });
        createHullShape(
            state,
            body,
            testShapeDef({ baseMaterial: { ...material, userMaterialId: BigInt(i + 1) } }),
            hull,
        );
        boxes.push({ body, position });
    }
    const rotation = { v: { x: 0, y: 0, z: 0 }, s: 1 };
    const velocity = { x: 0, y: 0, z: 0 };
    let tick = 0;
    return {
        step: () => {
            const before = mixes;
            velocity.x = tick++ % 2 === 0 ? 0.5 : -0.5;
            for (const box of boxes) {
                bodySetTransform(state, box.body, box.position, rotation);
                bodySetLinearVelocity(state, box.body, velocity);
            }
            world.step(1 / 60);
            if (awakeContactCount(state) !== boxes.length || mixes - before < boxes.length)
                throw new Error("allocation subject lost its awake custom-mixed contacts");
        },
        dispose: () => app.dispose(),
    };
}
