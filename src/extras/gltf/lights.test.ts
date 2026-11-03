import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { DirectionalLight, PointLight, SpotLight } from "../../core/rendering";
import { Transform } from "../../core/transform";
import { World } from "../../engine/ecs";
import { type GltfJson, parseGltf } from "./gltf";
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

test.each(
    (["directional", "point", "spot"] as const).flatMap((type) => [
        { type, hue: "neutral", channels: [0.0001, 0.0001, 0.0001] },
        { type, hue: "saturated", channels: [0.0001, 0, 0] },
        { type, hue: "black", channels: [0, 0, 0] },
    ]),
)("$type retains dim $hue illumination under sRGB hue encoding", async ({ type, channels }) => {
    const fixture = structuredClone(document);
    fixture.extensions!["KHR_lights_punctual"] = {
        lights: [{ type, color: channels, intensity: 1_000_000, range: 10, spot: {} }],
    };
    fixture.nodes = [{ extensions: { ["KHR_lights_punctual"]: { light: 0 } } }];
    fixture.animations = [];
    const world = new World();
    try {
        const gltf = await loadGltfScene(
            world,
            `data:application/json,${encodeURIComponent(JSON.stringify(fixture))}`,
        );
        const eid = spawnGltfScene(world, gltf).lightEntities[0];
        const component =
            type === "directional" ? DirectionalLight : type === "point" ? PointLight : SpotLight;
        const hex = world.storage(component).color.get(eid);
        const power =
            type === "directional"
                ? world.storage(DirectionalLight).illuminance.get(eid)
                : world.storage(type === "point" ? PointLight : SpotLight).intensity.get(eid) /
                  (4 * Math.PI);
        // At five metres on axis: extension candela / d², under Shallot's permitted range window.
        const attenuation = type === "directional" ? 1 : (1 - (5 / 10) ** 4) ** 2 / 5 ** 2;
        for (let channel = 0; channel < 3; channel++) {
            const encoded = ((hex >>> (16 - channel * 8)) & 255) / 255;
            const linear =
                encoded <= 0.04045 ? encoded / 12.92 : ((encoded + 0.055) / 1.055) ** 2.4;
            const expected = channels[channel] * 1_000_000 * attenuation;
            expect(linear * power * attenuation).toBeCloseTo(expected, 4);
        }
    } finally {
        world.dispose();
    }
});

test("punctual lights keep scene-node identity and ignore nodes outside the default scene", () => {
    const scene = parseGltf(document, buffers);
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
