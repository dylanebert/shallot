import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { gpuScene } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import { attachTexture, Camera, NormalPrepass } from "../../core/rendering";
import { viewTargetsKey } from "../../core/rendering/targets";
import { Transform } from "../../core/transform";
import { probeTexture } from "../../engine";
import {
    AlphaMode,
    lit,
    MaterialPlugin,
    MeshMaterial,
    MeshRenderPlugin,
    materialType,
    StandardRenderer,
    StandardRenderingPlugin,
} from "./index";

setDefaultTimeout(TIMEOUT.gpu);
const Parameters = d.struct({ color: d.vec4f });
const BentSix = materialType({
    name: "BentSix",
    parameters: Parameters,
    defaults: { color: d.vec4f(1) },
    depthPass: { prepass: true },
    varyings: { a: d.f32, b: d.f32, c: d.f32, e: d.f32, f: d.f32, g: d.f32 },
    vertex: (input) => {
        "use gpu";
        return {
            world: input.world,
            worldNormal: input.worldNormal,
            a: 1,
            b: 2,
            c: 3,
            e: 4,
            f: 5,
            g: 6,
        };
    },
    fragment: (ctx) => {
        "use gpu";
        const bend = (ctx.a + ctx.b * 2 + ctx.c * 3 + ctx.e * 4 + ctx.f * 5 + ctx.g * 6) / 91;
        return d.vec4f(
            lit(
                ctx.parameters.color.xyz,
                std.normalize(std.add(ctx.worldNormal, d.vec3f(0, bend, 0))),
            ),
            1,
        );
    },
});
const subject = gpuScene(
    import.meta.path,
    {
        defaults: false,
        plugins: [StandardRenderingPlugin, MeshRenderPlugin, MaterialPlugin(BentSix)],
    },
    (world) => {
        const mesh = world.create();
        world.add(mesh, Transform);
        world.add(mesh, MeshInstance);
        world.add(
            mesh,
            MeshMaterial,
            world.resource(BentSix).add({ color: d.vec4f(1) }, { alphaMode: AlphaMode.Mask(0.5) }),
        );
        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 0, 4, 0] });
        world.add(camera, Camera, { antialias: 0 });
        world.add(camera, StandardRenderer);
        world.add(camera, NormalPrepass);
        attachTexture(world, camera, { width: 33, height: 33 });
        return camera;
    },
);
test("the six-varying Mask prepass normal is the normal its fragment lit with", async () => {
    const { world, scene: camera } = subject();
    world.step(1 / 60);
    const target = world.resource(viewTargetsKey).normal.get(camera)!;
    const bytes = new DataView((await probeTexture(world, target.texture)).bytes);
    const packed = bytes.getUint32((16 * 33 + 16) * 4, true);
    const bent = [0, Math.SQRT1_2, Math.SQRT1_2];
    for (let c = 0; c < 3; c++) {
        const normal = (((packed >>> (c * 10)) & 1023) / 1023) * 2 - 1;
        expect(Math.abs(normal - bent[c]!)).toBeLessThan(2e-3);
    }
});
