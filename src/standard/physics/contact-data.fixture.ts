import {
    Body,
    BodyType,
    Compounds,
    Hulls,
    PhysicsMeshes,
    Shape,
    ShapeKind,
} from "../../core/physics";
import { Transform } from "../../core/transform";
import type { World } from "../../engine";

export function contactPair(world: World, scene: number): [number, number] {
    const ground = world.create();
    world.add(ground, Body);
    if (scene === 1) {
        const geometry = world.resource(PhysicsMeshes).add({
            name: "valley",
            vertices: new Float32Array([
                -2, 0.2, -2, -2, 0.2, 2, 0, 0, -2, 0, 0, 2, 2, 0.2, -2, 2, 0.2, 2,
            ]),
            indices: new Uint32Array([0, 1, 3, 3, 2, 0, 2, 3, 5, 5, 4, 2]),
            identifyEdges: true,
        });
        world.add(ground, Shape, { kind: ShapeKind.Mesh, geometry });
    } else if (scene === 2) {
        const geometry = world.resource(Compounds).add({
            name: "two spheres",
            spheres: [
                { center: [-1, 0, 0], radius: 0.5 },
                { center: [1, 0, 0], radius: 0.5 },
            ],
        });
        world.add(ground, Shape, { kind: ShapeKind.Compound, geometry });
    } else {
        world.add(ground, Shape, {
            geometry: world.resource(Hulls).add({ name: "ground", halfExtents: [4, 0.5, 4] }),
        });
    }
    const body = world.create();
    world.add(body, Transform, { translation: [0, scene === 1 ? 0.5 : 1, 0] });
    world.add(body, Body, { type: BodyType.Dynamic });
    world.add(body, Shape, {
        enableContactEvents: 1,
        geometry: world
            .resource(Hulls)
            .add({ name: "box", halfExtents: [scene === 2 ? 1.75 : 0.75, 0.5, 0.75] }),
    });
    return [ground, body];
}
