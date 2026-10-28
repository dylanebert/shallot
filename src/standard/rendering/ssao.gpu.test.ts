import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { Meshes, MeshInstance, sphere } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    computeClipFromWorld,
    DirectionalLight,
    GlobalAmbientLight,
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
/** [1-wide, 2-wide] cameras with SSAO, then the same pair without it. */
const narrow: number[] = [];
/** [with SSAO, without] over the ambient-occlusion example's cubes, at material occlusion 1, and
 * sphere, at 1.5: re-occluding with min(m, multibounce(1)) / m would light both alike. */
const exact: number[] = [];
let sun = 0;
function exactScene(world: World) {
    world.resource(GlobalAmbientLight).brightness = 80;
    const materials = world.resource(Materials);
    const cubes = materials.add(StandardMaterial({ baseColor: [0.1, 0.1, 0.1, 1] }));
    const ball = materials.add(StandardMaterial({ baseColor: [0.1, 0.1, 0.1, 1], occlusion: 1.5 }));
    for (const translation of [
        [0, 0, 1],
        [0, -1, 0],
        [1, 0, 0],
    ] as const) {
        const eid = world.create();
        world.add(eid, Transform, { translation: [...translation, 0] });
        world.add(eid, MeshInstance);
        world.add(eid, MeshMaterial, cubes);
    }
    const eid = world.create();
    world.add(eid, Transform);
    world.add(eid, MeshInstance, {
        mesh: world.resource(Meshes).add({ name: "Sphere", ...sphere(0.4, 72, 36) }),
    });
    world.add(eid, MeshMaterial, ball);
    for (const ssao of [true, false]) {
        const camera = world.create();
        world.add(camera, Transform, {
            translation: [2, 2, 2, 0],
            rotation: lookAtRotation(2, 2, 2, 0, 0, 0),
        });
        world.add(camera, Camera, { antialias: 0 });
        world.add(camera, StandardRenderer);
        if (ssao) world.add(camera, ScreenSpaceAmbientOcclusion);
        attachTexture(world, camera, { width: 256, height: 192 });
        exact.push(camera);
    }
}
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
    for (const ssao of [true, false]) {
        for (const width of [1, 2]) {
            const camera = world.create();
            world.add(camera, Transform, {
                translation: [0, 6, 3, 0],
                rotation: lookAtRotation(0, 6, 3, 0, 0, 0),
            });
            world.add(camera, Camera);
            world.add(camera, StandardRenderer);
            if (ssao) world.add(camera, ScreenSpaceAmbientOcclusion, { qualityLevel: 1 });
            attachTexture(world, camera, { width, height: 192 });
            narrow.push(camera);
        }
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
    {
        defaults: false,
        plugins: [
            StandardRenderingPlugin,
            MeshRenderPlugin,
            ScreenSpaceAmbientOcclusionPlugin,
            {
                name: "ExactAoScene",
                dependencies: [
                    StandardRenderingPlugin,
                    MeshRenderPlugin,
                    ScreenSpaceAmbientOcclusionPlugin,
                ],
                initialize: exactScene,
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
        const m = new Float32Array(16);
        computeClipFromWorld(world, eid, view.aspect, m);
        const w = m[3]! * x + m[11]! * z + m[15]!;
        const sx = (m[0]! * x + m[8]! * z + m[12]!) / w;
        const sy = (m[1]! * x + m[9]! * z + m[13]!) / w;
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
test("a 1-wide view occludes as a 2-wide one does, and a view without SSAO is unoccluded", async () => {
    const { world } = subjects()[0];
    world.step(1 / 60);
    async function color(eid: number) {
        const result = await probeTexture(world, offscreenTexture(world, eid)!);
        return new Float16Array(result.bytes);
    }
    const [one, two, plainOne, plainTwo] = await Promise.all(narrow.map(color));
    const luminance = (rgba: Float16Array, at: number) =>
        rgba[at]! * 0.2126 + rgba[at + 1]! * 0.7152 + rgba[at + 2]! * 0.0722;
    /** each row's least luminance ratio of `lit` to `plain` over its columns, and how many
     * channels differ and brighten. A 1-wide view's slices sample no columns beside its own, so it occludes
     * less than a 2-wide one, but no channel brightens. */
    function occlusion(lit: Float16Array, plain: Float16Array, width: number) {
        const ratios: number[] = [];
        for (let row = 0; row < 192; row++) {
            let ratio = Infinity;
            for (let column = 0; column < width; column++) {
                const at = (row * width + column) * 4;
                const unoccluded = luminance(plain, at);
                if (unoccluded > 0) ratio = Math.min(ratio, luminance(lit, at) / unoccluded);
            }
            ratios.push(ratio);
        }
        let changed = 0;
        let brightened = 0;
        for (let i = 0; i < lit.length; i++) {
            if (lit[i] !== plain[i]) changed++;
            if (lit[i]! > plain[i]!) brightened++;
        }
        return { ratios, changed, brightened };
    }
    const wide = occlusion(two!, plainTwo!, 2);
    const thin = occlusion(one!, plainOne!, 1);
    expect(Math.min(...wide.ratios)).toBeLessThan(0.8);
    expect(Math.min(...thin.ratios)).toBeLessThan(0.8);
    expect(thin.brightened).toBe(0);
    expect(thin.changed / one!.length).toBeGreaterThan((0.5 * wide.changed) / two!.length);
    world.remove(narrow[0]!, ScreenSpaceAmbientOcclusion);
    world.remove(narrow[1]!, ScreenSpaceAmbientOcclusion);
    world.step(1 / 60);
    const [stubOne, stubTwo, againOne, againTwo] = await Promise.all(narrow.map(color));
    expect(stubOne).toEqual(againOne!);
    expect(stubTwo).toEqual(againTwo!);
    expect(againOne).toEqual(plainOne!);
    world.add(narrow[0]!, ScreenSpaceAmbientOcclusion, { qualityLevel: 1 });
    world.add(narrow[1]!, ScreenSpaceAmbientOcclusion, { qualityLevel: 1 });
});
test("SSAO reading exactly 1, like its absence, leaves material occlusion whole, above 1 too", async () => {
    const { world } = subjects()[1];
    world.step(1 / 60);
    const [ssao, plain] = exact as [number, number];
    const ao = new Float32Array(
        (await probeTexture(world, world.resource(ssaoState).cameras.get(ssao)!.textures[4]!))
            .bytes,
    );
    const normal = new Uint32Array(
        (await probeTexture(world, world.resource(viewTargetsKey).normal.get(ssao)!.texture)).bytes,
    );
    const [lit, unlit] = await Promise.all(
        [ssao, plain].map(
            async (eid) =>
                new Float16Array((await probeTexture(world, offscreenTexture(world, eid)!)).bytes),
        ),
    );
    let unoccluded = 0;
    let occluded = 0;
    const differing: number[] = [];
    const reds = new Set<number>();
    for (let pixel = 0; pixel < ao.length; pixel++) {
        if (normal[pixel] === 0) continue;
        reds.add(unlit![pixel * 4]!);
        const same = [0, 1, 2, 3].every((c) => lit![pixel * 4 + c] === unlit![pixel * 4 + c]);
        if (ao[pixel] === 1) {
            unoccluded++;
            if (!same) differing.push(pixel);
        } else if (!same) occluded++;
    }
    // Ambient light alone shades each material flat, in proportion to its occlusion.
    const [dim, bright] = [...reds].sort((a, b) => a - b) as [number, number];
    expect(reds.size).toBe(2);
    expect(bright / dim).toBeCloseTo(1.5, 2);
    expect(unoccluded).toBeGreaterThan(0);
    expect(occluded).toBeGreaterThan(0);
    expect(differing).toEqual([]);
});
