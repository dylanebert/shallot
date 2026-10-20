import { beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    captureTexture,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import {
    extendedMaterial,
    MaterialPlugin,
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    materialType,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "./index";

setDefaultTimeout(TIMEOUT.gpu);
const PixelMaterial = materialType({
    name: "FramebufferPosition",
    parameters: d.struct({ unused: d.f32 }),
    defaults: { unused: 0 },
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(ctx.fragCoord.x / 8, ctx.fragCoord.y / 4, 0, 1);
    },
});
const ExtendedPixelMaterial = extendedMaterial(Materials, {
    name: "ExtendedFramebufferPosition",
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(ctx.fragCoord.x / 8, ctx.fragCoord.y / 4, 0, 1);
    },
});
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            StandardRenderingPlugin,
            MeshRenderPlugin,
            MaterialPlugin(PixelMaterial),
            MaterialPlugin(ExtendedPixelMaterial),
        ],
    },
]);
let camera: number;
let mesh: number;
beforeAll(() => {
    const { world } = subjects()[0];
    camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: 8, height: 4 });
    mesh = world.create();
    world.add(mesh, Transform, { scale: [20, 20, 1, 0] });
    world.add(mesh, MeshInstance);
    world.add(mesh, MeshMaterial, world.resource(PixelMaterial).add({ unused: 0 }));
    world.step(0);
    world.step(0);
}, TIMEOUT.startup);

function srgbByte(linear: number) {
    return Math.round((1.055 * linear ** (1 / 2.4) - 0.055) * 255);
}

for (const extended of [false, true]) {
    test(`${extended ? "extended" : "custom"} material reads top-left framebuffer pixel centres at target resolution`, async () => {
        const { world } = subjects()[0];
        world.remove(mesh, MeshMaterial);
        world.add(
            mesh,
            MeshMaterial,
            extended
                ? world.resource(ExtendedPixelMaterial).add(StandardMaterial())
                : world.resource(PixelMaterial).add({ unused: 0 }),
        );
        world.step(0);
        world.step(0);
        const { rgba } = await captureTexture(world, camera);
        for (let y = 0; y < 4; y++) {
            for (let x = 0; x < 8; x++) {
                const offset = (y * 8 + x) * 4;
                expect(Math.abs(rgba[offset] - srgbByte((x + 0.5) / 8))).toBeLessThanOrEqual(1);
                expect(Math.abs(rgba[offset + 1] - srgbByte((y + 0.5) / 4))).toBeLessThanOrEqual(1);
                expect(Array.from(rgba.subarray(offset + 2, offset + 4))).toEqual([0, 255]);
            }
        }
    });
}
