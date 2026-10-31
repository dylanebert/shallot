import { expect, spyOn, test } from "bun:test";
import { readFileSync } from "node:fs";
import { MeshPlugin } from "../../core/mesh";
import { GlobalTransform, TransformPlugin } from "../../core/transform";
import { openGpuApp as createApp } from "../../project/testing-gpu";
import type { GltfJson } from "./gltf";
import { loadGltfScene, poseGltfScene, spawnGltfScene } from "./scene";

function fixture(): GltfJson {
    return JSON.parse(
        readFileSync(new URL("./fixtures/skin-accessors.gltf", import.meta.url), "utf8"),
    );
}

for (const shear of [false, true])
    test(`skinned placement ignores the node global${shear ? " including shear" : ""}; rest bounds use linear blend joint globals`, async () => {
        const json = fixture();
        if (shear) {
            json.nodes![0].scale = [2, 1, 1];
            json.nodes![6].rotation = [0, 0, Math.sin(Math.PI / 8), Math.cos(Math.PI / 8)];
        }
        const original = globalThis.fetch;
        const fetch = spyOn(globalThis, "fetch").mockImplementation(((input, init) =>
            String(input) === "https://fixture.test/skin.gltf"
                ? Promise.resolve(Response.json(json))
                : original(input, init)) as typeof globalThis.fetch);
        const app = await createApp({ defaults: false, plugins: [MeshPlugin, TransformPlugin] });
        try {
            const gltf = await loadGltfScene(app.world, "https://fixture.test/skin.gltf");
            expect(gltf.skins).toHaveLength(4);
            expect(gltf.skeletons).toHaveLength(2);
            // The ignored translations are 100 and 200. Joint A rests at (2,0,0),
            // B at (2,3,0), C at (2,6,0); identity inverse binds leave these globals.
            const sx = shear ? 2 : 1;
            expect(gltf.bounds.min).toEqual([0, 0, 0]);
            expect(gltf.bounds.max[0]).toBe(3 * sx);
            expect(gltf.bounds.max[1]).toBe(7);
            const instance = spawnGltfScene(app.world, gltf, {
                translation: [7, 8, 9],
                scale: [2, 1, 1],
            });
            app.world.tick();
            const global = app.world.storage(GlobalTransform);
            for (const eid of instance.entities) {
                expect([
                    global.translation.x.get(eid),
                    global.translation.y.get(eid),
                    global.translation.z.get(eid),
                ]).toEqual([7, 8, 9]);
                expect([
                    global.scale.x.get(eid),
                    global.scale.y.get(eid),
                    global.scale.z.get(eid),
                ]).toEqual([2, 1, 1]);
            }
            poseGltfScene(app.world, instance, { name: "rest", samplers: [], channels: [] }, 0);
            app.world.tick();
            for (const eid of instance.entities)
                expect([
                    global.translation.x.get(eid),
                    global.translation.y.get(eid),
                    global.translation.z.get(eid),
                ]).toEqual([7, 8, 9]);
            expect(instance.globals[6][12]).toBe(100 * sx);
            if (shear) {
                const unskinned = {
                    ...gltf,
                    primitives: gltf.primitives.map((primitive) => ({ ...primitive, skin: -1 })),
                };
                expect(() => spawnGltfScene(app.world, unskinned)).toThrow(/Unorm8.*shear/);
            }
            expect(gltf.primitives[0].matrix[12]).toBe(100 * sx);
        } finally {
            app.dispose();
            fetch.mockRestore();
        }
    });
