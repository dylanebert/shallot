import { expect, test } from "bun:test";
import { Body, Hulls, PhysicsPlugin, Shape } from "../../src/core/physics";
import { TransformPlugin } from "../../src/core/transform";
import { World } from "../../src/engine";
import { StandardPhysicsPlugin } from "../../src/standard/physics";
import { physicsState } from "../../src/standard/physics/runtime";
import { hullImage } from "../../src/standard/physics/shapes/hullbytes";
import { readShapeHull } from "../../src/standard/physics/shapes/hull";
import { nativeBinary, run } from "./native";

const SCALES = [
    [1, 1, 1],
    [-0.75, 1, 1],
    [1.2, 1, -0.9],
    [0.9, 0.9, 0.9],
    [1.25, 0.75, 1.5],
] as const;
const H = Math.fround(Math.SQRT1_2);

test("a placed hull Shape's kernel hull equals b3CreateTransformedHullShape's at every Shape.scale", async () => {
    const native = JSON.parse(run([nativeBinary("hull-shape.c")])) as number[][];
    const world = new World();
    for (const component of [...TransformPlugin.components!, ...PhysicsPlugin.components!])
        world.registry.register(component, PhysicsPlugin.name);
    await StandardPhysicsPlugin.initialize!(world);
    await StandardPhysicsPlugin.warm!(world);
    for (const system of StandardPhysicsPlugin.systems!) world.addSystem(system, StandardPhysicsPlugin.name);
    try {
        const hulls = world.resource(Hulls);
        const sources = [
            hulls.add({ name: "box", halfExtents: [0.4, 0.6, 0.5] }),
            hulls.add({ name: "cylinder", cylinder: { height: 1, radius: 0.5, yOffset: 0, sides: 15 } }),
            hulls.add({ name: "tetra", points: new Float32Array([0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4]), maxVertexCount: 4 }),
        ];
        const labels: string[] = [];
        for (const [source, geometry] of sources.entries())
            for (const scale of SCALES) {
                const eid = world.create();
                world.add(eid, Body);
                world.add(eid, Shape, {
                    geometry,
                    scale: [...scale],
                    hullPosition: [2, -3, 4],
                    hullRotation: [0, 0, H, H],
                });
                labels.push(`${hulls.get(sources[source]!)!.name} at ${scale.join(", ")}`);
            }
        world.tick();
        const state = physicsState(world)!;
        for (const [shape, label] of labels.entries()) {
            const image = hullImage(readShapeHull(state, shape));
            expect(Array.from(new Uint32Array(image.buffer, image.byteOffset, image.length / 4)), label).toEqual(native[shape]!);
        }
    } finally {
        await StandardPhysicsPlugin.dispose!(world);
        world.dispose();
    }
});
