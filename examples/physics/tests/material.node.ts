import { expect, setDefaultTimeout, test } from "bun:test";
import {
    BaseColor,
    Body,
    BodyType,
    Camera,
    DirectionalLight,
    MeshInstance,
    MeshMaterial,
    Shape,
    ShapeKind,
    Sleeping,
    StandardMaterial,
    StandardRenderer,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { attachTexture, captureTexture } from "@dylanebert/shallot/rendering";
import { StandardPhysicsPlugin, setAwake } from "@dylanebert/shallot/standard/physics";
import {
    Materials,
    MeshRenderPlugin,
    StandardRenderingPlugin,
} from "@dylanebert/shallot/standard/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import * as d from "typegpu/data";
import { ShapeMaterial, ShapeMaterialPlugin } from "../src/material";

setDefaultTimeout(TIMEOUT.node);
function createScene(world: World) {
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 9] });
    world.add(camera, Camera, { antialias: 0 });
    world.add(camera, StandardRenderer);
    attachTexture(world, camera, { width: 256, height: 96 });
    const sun = world.create();
    world.add(sun, Transform);
    world.add(sun, DirectionalLight, { shadowMapsEnabled: 0 });
    const subject = world.resource(ShapeMaterial).add({
        ...StandardMaterial({ perceptualRoughness: 0.38 }),
        restingColor: d.vec3f(0.52, 0.29, 0.065),
    });
    const teal = world.resource(ShapeMaterial).add({
        ...StandardMaterial({ perceptualRoughness: 0.38 }),
        restingColor: d.vec3f(0.055, 0.29, 0.35),
    });
    const colors = [
        [0.72, 0.4, 0.05, 1],
        [0.04, 0.43, 0.53, 1],
        [1, 0.22, 0.035, 1],
        [1, 0.22, 0.035, 1],
    ] as const;
    const resting = [
        [0.52, 0.29, 0.065, 1],
        [0.055, 0.29, 0.35, 1],
        colors[2],
        [0.65, 0.16, 0.045, 1],
    ] as const;
    const shapes = colors.map((color, i) => {
        const body = world.create();
        world.add(body, Transform, { translation: [(i - 1.5) * 1.5, 0, 0] });
        world.add(body, Body, {
            type: BodyType.Dynamic,
            gravityScale: 0,
            isAwake: Number(i === 2),
        });
        const shape = world.create();
        world.add(shape, Shape, { body, kind: ShapeKind.Hull });
        world.add(shape, MeshInstance);
        world.add(shape, MeshMaterial, i === 1 ? teal : subject);
        world.add(shape, BaseColor, { color });
        return { body, shape };
    });
    const references = resting.map((baseColor) =>
        world.resource(Materials).add(StandardMaterial({ baseColor, perceptualRoughness: 0.38 })),
    );
    return { camera, shapes, references, subject, teal, colors };
}

test("attached shapes shade sleep and reactions as base color only, sharing their role material", async () => {
    let scene!: ReturnType<typeof createScene>;
    const app = await openGpuApp({
        defaults: false,
        plugins: [
            {
                name: "ShapeMaterialTest",
                dependencies: [
                    StandardRenderingPlugin,
                    MeshRenderPlugin,
                    StandardPhysicsPlugin,
                    ShapeMaterialPlugin,
                ],
                initialize(world) {
                    scene = createScene(world);
                },
            },
        ],
    });
    try {
        const world = app.world;
        world.tick();
        world.step(0);
        const varied = (await captureTexture(world, scene.camera)).rgba;
        for (const [i, { shape }] of scene.shapes.entries()) {
            expect(world.has(shape, Sleeping)).toBe(i !== 2);
            world.remove(shape, BaseColor);
            world.add(shape, MeshMaterial, scene.references[i]!);
        }
        world.step(0);
        expect((await captureTexture(world, scene.camera)).rgba).toEqual(varied);
        expect(new Set(varied).size).toBeGreaterThan(4);
        for (const [i, { body, shape }] of scene.shapes.entries()) {
            world.add(shape, BaseColor, { color: scene.colors[i]! });
            world.add(shape, MeshMaterial, i === 1 ? scene.teal : scene.subject);
            setAwake(world, body, true);
        }
        world.tick();
        world.step(0);
        expect((await captureTexture(world, scene.camera)).rgba).not.toEqual(varied);
        for (const { shape } of scene.shapes) expect(world.has(shape, Sleeping)).toBe(false);
    } finally {
        app.dispose();
    }
});
