import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { readFileSync } from "node:fs";
import {
    composeMat4,
    createApp,
    DirectionalLight,
    GlobalTransform,
    MeshPlugin,
    PointLight,
    SpotLight,
    Transform,
    TransformPlugin,
} from "@dylanebert/shallot";
import { loadGltfScene, poseGltfScene, spawnGltfScene } from "@dylanebert/shallot/gltf";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import gold from "./lights.gold.json";
import { assets, fileFetch, syncAssets } from "./setup";

setDefaultTimeout(TIMEOUT.node);
await syncAssets();
await setupGlobals();

for (const record of gold.records) {
    test(`${record.name} light photometry and posed world placements match KHR_lights_punctual and recorded node matrices`, async () => {
        const path = new URL(
            record.name === "PunctualLights"
                ? "../models/PunctualLights.gltf"
                : `Models/${record.name}/glTF/${record.name}.gltf`,
            assets,
        );
        const json = JSON.parse(readFileSync(path, "utf8"));
        const fetch = spyOn(globalThis, "fetch").mockImplementation(fileFetch);
        const app = await createApp({
            defaults: false,
            plugins: [
                MeshPlugin,
                TransformPlugin,
                {
                    name: "LitScene",
                    async initialize(world) {
                        const gltf = await loadGltfScene(world, path);
                        expect(gltf.unsupported).not.toContain("KHR_lights_punctual");
                        const instance = spawnGltfScene(world, gltf);
                        expect(instance.lightEntities.length).toBeGreaterThan(0);
                        const lights = json.extensions.KHR_lights_punctual.lights;
                        for (let i = 0; i < instance.lightEntities.length; i++) {
                            const eid = instance.lightEntities[i];
                            const node = gltf.nodes[gltf.lights[i].nodeIndex].index;
                            const light =
                                lights[json.nodes[node].extensions.KHR_lights_punctual.light];
                            const component =
                                light.type === "directional"
                                    ? DirectionalLight
                                    : light.type === "point"
                                      ? PointLight
                                      : SpotLight;
                            expect(world.has(eid, component)).toBe(true);
                            if (light.type === "directional")
                                expect(world.storage(DirectionalLight).illuminance.get(eid)).toBe(
                                    Math.fround(light.intensity ?? 1),
                                );
                            else {
                                const storage = world.storage(
                                    light.type === "point" ? PointLight : SpotLight,
                                );
                                expect(storage.intensity.get(eid)).toBe(
                                    Math.fround((light.intensity ?? 1) * 4 * Math.PI),
                                );
                                expect(storage.range.get(eid)).toBe(
                                    Math.fround(
                                        light.range ??
                                            Math.max(1, Math.sqrt((light.intensity ?? 1) / 0.01)),
                                    ),
                                );
                                expect(storage.radius.get(eid)).toBe(0);
                                if (light.type === "spot") {
                                    expect(world.storage(SpotLight).innerAngle.get(eid)).toBe(
                                        Math.fround(light.spot.innerConeAngle ?? 0),
                                    );
                                    expect(world.storage(SpotLight).outerAngle.get(eid)).toBe(
                                        Math.fround(light.spot.outerConeAngle ?? Math.PI / 4),
                                    );
                                }
                            }
                            // IEC 61966-2-1: component colors are sRGB hex, not glTF's linear multipliers.
                            const color = (light.color ?? [1, 1, 1]).reduce(
                                (hex: number, c: number) =>
                                    (hex << 8) |
                                    Math.round(
                                        255 *
                                            (c <= 0.0031308
                                                ? c * 12.92
                                                : 1.055 * c ** (1 / 2.4) - 0.055),
                                    ),
                                0,
                            );
                            expect(world.storage(component).color.get(eid)).toBe(color);
                        }
                        const samples = [
                            ...[-1, 0, 0.5, 1, 2].map((time) => ({
                                animation: -1,
                                time,
                                worlds: record.nodes,
                            })),
                            ...record.animations.flatMap((animation) =>
                                animation.samples.map((sample) => ({
                                    animation: animation.index,
                                    ...sample,
                                })),
                            ),
                        ];
                        for (const sample of samples) {
                            poseGltfScene(
                                world,
                                instance,
                                gltf.animations[sample.animation] ?? {
                                    name: "rest",
                                    channels: [],
                                    samplers: [],
                                },
                                sample.time,
                            );
                            world.tick();
                            const storage = world.storage(GlobalTransform);
                            for (const [eid, node] of [
                                ...instance.entities.map((eid, i) => [
                                    eid,
                                    gltf.nodes[gltf.primitives[i].nodeIndex].index,
                                ]),
                                ...instance.lightEntities.map((eid, i) => [
                                    eid,
                                    gltf.nodes[gltf.lights[i].nodeIndex].index,
                                ]),
                            ]) {
                                const expected = sample.worlds.find(
                                    (entry) => entry.node === node,
                                )!.matrix;
                                const matrix = composeMat4(
                                    storage.translation.x.get(eid),
                                    storage.translation.y.get(eid),
                                    storage.translation.z.get(eid),
                                    storage.rotation.x.get(eid),
                                    storage.rotation.y.get(eid),
                                    storage.rotation.z.get(eid),
                                    storage.rotation.w.get(eid),
                                    storage.scale.x.get(eid),
                                    storage.scale.y.get(eid),
                                    storage.scale.z.get(eid),
                                );
                                for (let axis = 0; axis < 16; axis++)
                                    expect(
                                        Math.abs(matrix[axis] - expected[axis]),
                                    ).toBeLessThanOrEqual(
                                        3e-5 * Math.max(1, Math.abs(expected[axis])),
                                    );
                            }
                        }
                        // Imported lights survive as ordinary independently writable entities.
                        expect(world.has(instance.lightEntities[0], Transform)).toBe(true);
                    },
                },
            ],
        }).finally(() => fetch.mockRestore());
        try {
            console.log("lit glTF parity", record.name, app.world.gpu.adapter);
        } finally {
            app.dispose();
        }
    });
}
