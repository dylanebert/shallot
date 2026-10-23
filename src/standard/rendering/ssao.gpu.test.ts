import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    DirectionalLight,
    GlobalAmbientLight,
    RenderContext,
    Views,
} from "../../core/rendering";
import { viewTargetsKey } from "../../core/rendering/targets";
import { offscreenTexture } from "../../core/rendering/view";
import { Transform } from "../../core/transform";
import { lookAtRotation, probeTexture, type World } from "../../engine";
import { StandardRenderer, StandardRenderingPlugin } from "./forward";
import { Materials, MeshMaterial, StandardMaterial } from "./material";
import { MeshRenderPlugin } from "./mesh-render";
import { ScreenSpaceAmbientOcclusion, ScreenSpaceAmbientOcclusionPlugin, ssaoState } from "./ssao";

setDefaultTimeout(TIMEOUT.gpu);
const cameras: number[] = [];
let sun = 0;
function scene(world: World) {
    const material = world
        .resource(Materials)
        .add(StandardMaterial({ baseColor: [0.1, 0.1, 0.1, 1], perceptualRoughness: 1 }));
    world.resource(GlobalAmbientLight).brightness = 80;
    sun = world.create();
    world.add(sun, Transform, { rotation: [-Math.SQRT1_2, 0, 0, Math.SQRT1_2] });
    world.add(sun, DirectionalLight, { illuminance: 0 });
    const ground = world.create();
    world.add(ground, Transform, { translation: [0, -0.1, 0, 0], scale: [12, 0.2, 12, 0] });
    world.add(ground, MeshInstance);
    world.add(ground, MeshMaterial, material);
    const box = world.create();
    world.add(box, Transform, { translation: [0, 1, 0, 0], scale: [2, 2, 2, 0] });
    world.add(box, MeshInstance);
    world.add(box, MeshMaterial, material);
    for (let aa = 0; aa < 2; aa++) {
        const camera = world.create();
        world.add(camera, Transform, {
            translation: [0, 6, 3, 0],
            rotation: lookAtRotation(0, 6, 3, 0, 0, 0),
        });
        world.add(camera, Camera, { antialias: aa });
        world.add(camera, StandardRenderer);
        world.add(camera, ScreenSpaceAmbientOcclusion, {
            qualityLevel: 1,
            constantObjectThickness: 2,
        });
        attachTexture(world, camera, { width: 256, height: 192 });
        cameras.push(camera);
    }
}
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            StandardRenderingPlugin,
            MeshRenderPlugin,
            ScreenSpaceAmbientOcclusionPlugin,
            {
                name: "AoScene",
                dependencies: [
                    StandardRenderingPlugin,
                    MeshRenderPlugin,
                    ScreenSpaceAmbientOcclusionPlugin,
                ],
                initialize: scene,
            },
        ],
    },
]);

test("Medium occludes the ambient-lit plane at box contact, and absent SSAO leaves it uniform", async () => {
    const { world } = subjects()[0];
    world.step(1 / 60);
    const eid = cameras[0]!;
    async function luminance(x: number, z: number) {
        const view = world.resource(Views).get(eid)!;
        const m = world.resource(RenderContext).viewStaging;
        const at = view.slot * 64;
        const w = m[at + 3]! * x + m[at + 11]! * z + m[at + 15]!;
        const sx = (m[at]! * x + m[at + 8]! * z + m[at + 12]!) / w;
        const sy = (m[at + 1]! * x + m[at + 9]! * z + m[at + 13]!) / w;
        const px = Math.floor((sx * 0.5 + 0.5) * view.width);
        const py = Math.ceil((0.5 - sy * 0.5) * view.height);
        const result = await probeTexture(world, offscreenTexture(world, eid)!, {
            origin: [px, py],
            size: [1, 1],
        });
        const target = world.resource(viewTargetsKey).normal.get(eid);
        if (target) {
            const normal = await probeTexture(world, target.texture, {
                origin: [px, py],
                size: [1, 1],
            });
            const packed = new DataView(normal.bytes).getUint32(0, true);
            expect((((packed >>> 10) & 1023) / 1023) * 2 - 1).toBeGreaterThan(0.99);
        }
        const rgba = new Float16Array(result.bytes);
        return rgba[0]! * 0.2126 + rgba[1]! * 0.7152 + rgba[2]! * 0.0722;
    }
    const contact = await luminance(0, 1.001);
    const far = await luminance(4, 1.001);
    expect(contact).toBeLessThan(0.8 * far);
    world.remove(eid, ScreenSpaceAmbientOcclusion);
    world.step(1 / 60);
    const absentContact = await luminance(0, 1.001);
    const absentFar = await luminance(4, 1.001);
    expect(absentFar).toBeGreaterThan(0);
    expect(Math.abs(absentContact - absentFar)).toBeLessThanOrEqual(1e-3);
    world.add(eid, ScreenSpaceAmbientOcclusion, { qualityLevel: 1 });
    world.resource(GlobalAmbientLight).brightness = 0;
    world.storage(DirectionalLight).illuminance.set(sun, 10000);
    world.step(1 / 60);
    const directWithAo = await luminance(0, 1.001);
    world.remove(eid, ScreenSpaceAmbientOcclusion);
    world.step(1 / 60);
    const directWithoutAo = await luminance(0, 1.001);
    expect(directWithoutAo).toBeGreaterThan(0);
    expect(directWithAo).toBe(directWithoutAo);
    world.resource(GlobalAmbientLight).brightness = 80;
    world.storage(DirectionalLight).illuminance.set(sun, 0);
    world.add(eid, ScreenSpaceAmbientOcclusion, { qualityLevel: 1 });
});
test("every quality runs with and without MSAA, including Custom", async () => {
    const { world } = subjects()[0];
    for (let quality = 0; quality < 5; quality++) {
        for (const eid of cameras) {
            const columns = world.storage(ScreenSpaceAmbientOcclusion);
            columns.qualityLevel.set(eid, quality);
            columns.sliceCount.set(eid, 5);
            columns.samplesPerSliceSide.set(eid, 4);
        }
        world.step(1 / 60);
        for (const eid of cameras) {
            const camera = world.resource(ssaoState).cameras.get(eid)!;
            const result = await probeTexture(world, camera.textures[4]!);
            const values = new Float32Array(result.bytes);
            expect(values.length).toBe(256 * 192);
            expect(
                values.every(
                    (value) => Number.isFinite(value) && value >= 0.02999 && value <= 1.00001,
                ),
            ).toBe(true);
            expect(world.resource(Views).get(eid)!.normal).not.toBeNull();
        }
    }
});
