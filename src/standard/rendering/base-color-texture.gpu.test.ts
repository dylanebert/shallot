import { expect, setDefaultTimeout, test } from "bun:test";
import { registerImage } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { Meshes, MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    captureTexture,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { ImageBindings } from "../../core/rendering/image";
import { Transform } from "../../core/transform";
import {
    AlphaMode,
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "./index";

setDefaultTimeout(TIMEOUT.gpu);
let imageId = 0;
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            StandardRenderingPlugin,
            MeshRenderPlugin,
            {
                name: "TwoColourImage",
                initialize(world) {
                    const png =
                        "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAYAAACp8Z5+AAAAEklEQVR4nGP4z8DwH4SRIKkCADRIH+EDrgQAAAAAAElFTkSuQmCC";
                    imageId = registerImage(
                        world,
                        new Blob([Uint8Array.from(atob(png), (c) => c.charCodeAt(0))], {
                            type: "image/png",
                        }),
                    );
                },
            },
        ],
    },
]);

test("StandardMaterial samples both image halves at mesh UVs and multiplies baseColor RGBA", async () => {
    const { world } = subjects()[0];
    const device = world.gpu.device;
    // Dawn has no image decoder; upload the image pixels directly to isolate material sampling.
    const image = device.createTexture({
        size: [4, 4, 1],
        format: "rgba8unorm-srgb",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    world.own(image);
    const pixels = new Uint8Array(4 * 4 * 4);
    for (let y = 0; y < 4; y++) {
        for (let x = 0; x < 4; x++)
            pixels.set(x < 2 ? [255, 0, 0, 255] : [0, 255, 0, 255], (y * 4 + x) * 4);
    }
    device.queue.writeTexture({ texture: image }, pixels, { bytesPerRow: 16 }, [4, 4, 1]);
    world.resource(ImageBindings).images = image;
    const mesh = world.resource(Meshes).add({
        name: "two-colour-quad",
        vertices: new Float32Array([
            -2, -2, 0, 0, 0, 0, 1, 0, 2, -2, 0, 1, 0, 0, 1, 0, 2, 2, 0, 1, 0, 0, 1, 1, -2, 2, 0, 0,
            0, 0, 1, 1,
        ]),
        indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    });
    const materials = world.resource(Materials);
    const row = materials.add(
        StandardMaterial({
            baseColorTexture: imageId,
            unlit: true,
        }),
    );
    const eid = world.create();
    world.add(eid, Transform);
    world.add(eid, MeshInstance, { mesh });
    world.add(eid, MeshMaterial, row);
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.storage(Camera).antialias.set(camera, 0);
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: 32, height: 32 });
    for (const [factor, byte] of [
        [1, 255],
        [0.5, 188],
    ]) {
        materials.update(row, { baseColor: [factor, factor, factor, factor] });
        world.step(0);
        world.step(0);
        const { rgba } = await captureTexture(world, camera);
        const pixel = (x: number) =>
            Array.from(rgba.subarray((16 * 32 + x) * 4, (16 * 32 + x) * 4 + 4));
        expect(pixel(9)).toEqual([byte, 0, 0, 255]);
        expect(pixel(22)).toEqual([0, byte, 0, 255]);
    }
    for (let y = 0; y < 4; y++) {
        for (let x = 0; x < 2; x++) pixels[(y * 4 + x) * 4 + 3] = 128;
    }
    device.queue.writeTexture({ texture: image }, pixels, { bytesPerRow: 16 }, [4, 4, 1]);
    materials.update(row, {}, { alphaMode: AlphaMode.Mask(0.375) });
    world.step(0);
    world.step(0);
    const { rgba } = await captureTexture(world, camera);
    expect(Array.from(rgba.subarray((16 * 32 + 9) * 4, (16 * 32 + 9) * 4 + 4))).toEqual(
        Array.from(rgba.subarray(0, 4)),
    );
    expect(Array.from(rgba.subarray((16 * 32 + 22) * 4, (16 * 32 + 22) * 4 + 4))).toEqual([
        0, 188, 0, 255,
    ]);
});
