import { createApp, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { physicsWorld } from "@dylanebert/shallot/standard/physics";
import type { Body } from "../api/body";
import type { Vec3 } from "../common/math";
import { BodyType, defaultSurfaceMaterial } from "../common/types";
import { createCompound } from "../shapes/compound";
import { makeBoxHull } from "../shapes/hull";
import { createMesh } from "../shapes/mesh";
import { defaultFrictionCallback, defaultRestitutionCallback } from "../world/world";
import { awakeContactCount } from "./contact";

export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};

export default async function create(kind: string) {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const world = app.world;
    const physics = physicsWorld(world)!;
    const state = physics.state;
    state.enableSleep = false;
    state.contactRecycleDistance = 0;
    let mixes = 0;
    state.frictionCallback = (a, ia, b, ib) => {
        mixes++;
        return defaultFrictionCallback(a, ia, b, ib);
    };
    state.restitutionCallback = (a, ia, b, ib) => defaultRestitutionCallback(a, ia, b, ib);
    const material = { ...defaultSurfaceMaterial(), userMaterialId: 0x123456789abcdef0n };
    const ground = physics.createBody({ type: BodyType.Static });
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
        ground.createMesh({ baseMaterial: material }, mesh, { x: 1, y: 1, z: 1 });
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
        ground.createCompound({}, compound);
    }
    const hull = makeBoxHull(0.5, 0.5, 0.5);
    const boxes: { body: Body; position: Vec3 }[] = [];
    for (let i = 0; i < 64; i++) {
        const position = { x: (i % 8) * 3 - 12, y: 0.5, z: Math.floor(i / 8) * 3 - 12 };
        const body = physics.createBody({ type: BodyType.Dynamic, position });
        body.createHull({ baseMaterial: { ...material, userMaterialId: BigInt(i + 1) } }, hull);
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
                box.body.setTransform(box.position, rotation);
                box.body.setLinearVelocity(velocity);
            }
            world.step(1 / 60);
            if (awakeContactCount(state) !== boxes.length || mixes - before < boxes.length)
                throw new Error("allocation subject lost its awake custom-mixed contacts");
        },
        dispose: () => app.dispose(),
    };
}
