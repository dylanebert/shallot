import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { DirectionalLight, PointLight, SpotLight } from "../../core/rendering";
import { Transform } from "../../core/transform";
import { World } from "../../engine/ecs";
import { type GltfJson, parse } from "./gltf";
import { loadGltfScene, poseGltfScene, spawnGltfScene } from "./scene";

const document: GltfJson = JSON.parse(
    readFileSync(
        new URL("../../../examples/gltf/public/models/PunctualLights.gltf", import.meta.url),
        "utf8",
    ),
);
const buffers = document.buffers!.map(
    ({ uri }) => Uint8Array.from(Buffer.from(uri!.split(",")[1], "base64")).buffer,
);

test("punctual lights keep scene-node identity and ignore nodes outside the default scene", () => {
    const scene = parse(document, buffers);
    expect(scene.unsupported).toEqual([]);
    expect(scene.lights.map((light) => light.type)).toEqual([
        "directional",
        "point",
        "spot",
        "spot",
    ]);
    expect(scene.lights.map((light) => light.nodeIndex)).toEqual([1, 2, 3, 4]);
    for (const light of scene.lights)
        expect(Array.from(light.matrix.slice(12, 15))).toEqual([4, 3, 4]);
});

test("punctual lights convert photometry, preserve half-angles and follow their posed nodes", async () => {
    const world = new World();
    try {
        const gltf = await loadGltfScene(
            world,
            `data:application/json,${encodeURIComponent(JSON.stringify(document))}`,
        );
        const instance = spawnGltfScene(world, gltf, { translation: [10, 0, 0], scale: [3, 3, 3] });
        expect(instance.entities).toHaveLength(0);
        expect(instance.lightEntities).toHaveLength(4);
        const [sun, point, spot, defaults] = instance.lightEntities;
        expect(world.has(sun, DirectionalLight)).toBe(true);
        expect(world.has(point, PointLight)).toBe(true);
        expect(world.has(spot, SpotLight)).toBe(true);
        expect(world.storage(DirectionalLight).illuminance.get(sun)).toBe(123);
        expect(world.storage(DirectionalLight).color.get(sun)).toBe(0xbc89ff);
        expect(world.storage(PointLight).intensity.get(point)).toBeCloseTo(7 * 4 * Math.PI, 4);
        expect(world.storage(PointLight).range.get(point)).toBe(11);
        const spots = world.storage(SpotLight);
        expect(spots.intensity.get(spot)).toBeCloseTo(9 * 4 * Math.PI, 4);
        expect(spots.range.get(spot)).toBe(30);
        expect(spots.innerAngle.get(spot)).toBeCloseTo(0.2);
        expect(spots.outerAngle.get(spot)).toBeCloseTo(0.6);
        expect(spots.innerAngle.get(defaults)).toBe(0);
        expect(spots.outerAngle.get(defaults)).toBeCloseTo(Math.PI / 4);
        expect(spots.intensity.get(defaults)).toBeCloseTo(4 * Math.PI);
        expect(spots.range.get(defaults)).toBe(10);
        expect(spots.color.get(defaults)).toBe(0xffffff);
        expect(spots.radius.get(spot)).toBe(0);
        const transforms = world.storage(Transform);
        for (const eid of instance.lightEntities) {
            expect(transforms.translation.x.get(eid)).toBe(22);
            expect(transforms.translation.y.get(eid)).toBe(9);
            expect(transforms.translation.z.get(eid)).toBe(12);
            expect(transforms.rotation.y.get(eid)).toBeCloseTo(Math.SQRT1_2);
        }
        poseGltfScene(world, instance, gltf.animations[0], 0.5);
        for (const eid of instance.lightEntities) {
            expect(transforms.translation.x.get(eid)).toBe(25);
            expect(transforms.translation.y.get(eid)).toBe(12);
            expect(transforms.translation.z.get(eid)).toBe(15);
        }
        expect(spots.range.get(spot)).toBe(30);
        expect(spots.intensity.get(spot)).toBeCloseTo(9 * 4 * Math.PI, 4);
        poseGltfScene(world, instance, gltf.animations[1], 0.5);
        for (const eid of instance.lightEntities) {
            expect(transforms.translation.x.get(eid)).toBeCloseTo(10 + 3 * (2 + Math.SQRT2), 4);
            expect(transforms.translation.z.get(eid)).toBeCloseTo(3 * (4 - Math.SQRT2), 4);
            expect(transforms.rotation.y.get(eid)).toBeCloseTo(Math.sin((3 * Math.PI) / 8), 5);
            expect(transforms.rotation.w.get(eid)).toBeCloseTo(Math.cos((3 * Math.PI) / 8), 5);
        }
    } finally {
        world.dispose();
    }
});
