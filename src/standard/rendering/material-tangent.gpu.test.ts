import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { gpuApps } from "../../../scripts/gpu.fixture";
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
const Tangent = materialType({
    name: "WorldTangentProbe",
    parameters: d.struct({ unused: d.u32 }),
    defaults: { unused: 0 },
    vertexTangent: true,
    fragmentInputs: { tangent: true },
    depthPass: { prepass: true, shadows: true },
    varyings: { tint: d.vec3f },
    vertex: (input) => {
        "use gpu";
        return {
            world: input.world,
            worldNormal: input.worldNormal,
            worldTangent: d.vec4f(0, 2, 0, -1),
            tint: d.vec3f(1),
        };
    },
    fragment: (ctx) => {
        "use gpu";
        const normal = std.normalize(d.vec3f(ctx.tangent.xy, ctx.tangent.w));
        return d.vec4f(lit(ctx.tint, normal), 1);
    },
});
const TangentPlugin = MaterialPlugin(Tangent);
const subjects: { camera: number; mirrored: boolean; masked: boolean }[] = [];
const apps = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            StandardRenderingPlugin,
            MeshRenderPlugin,
            TangentPlugin,
            {
                name: "TangentScene",
                dependencies: [StandardRenderingPlugin, MeshRenderPlugin, TangentPlugin],
                initialize(world) {
                    for (const masked of [false, true])
                        for (const mirrored of [false, true]) {
                            const x = subjects.length * 10;
                            const entity = world.create();
                            world.add(entity, Transform, {
                                translation: [x, 0, 0],
                                scale: [mirrored ? -1 : 1, 1, 1],
                            });
                            // The default cube has no tangent stream: the vertex function supplies the entire tangent.
                            world.add(entity, MeshInstance);
                            world.add(
                                entity,
                                MeshMaterial,
                                world.resource(Tangent).add(
                                    { unused: 0 },
                                    {
                                        alphaMode: masked ? AlphaMode.Mask(0.5) : AlphaMode.Opaque,
                                        cullMode: "none",
                                    },
                                ),
                            );
                            const camera = world.create();
                            world.add(camera, Transform, { translation: [x, 0, 4] });
                            world.add(camera, Camera, { antialias: 0 });
                            world.add(camera, StandardRenderer);
                            world.add(camera, NormalPrepass);
                            attachTexture(world, camera, { width: 33, height: 33 });
                            subjects.push({ camera, mirrored, masked });
                        }
                },
            },
        ],
    },
]);
for (const masked of [false, true])
    for (const mirrored of [false, true])
        test(`world tangent without a mesh stream: ${masked ? "masked" : "opaque"} ${mirrored ? "mirrored" : "ordinary"} normal prepass normalizes xyz and corrects handedness`, async () => {
            const { world } = apps()[0];
            world.step(1 / 60);
            const subject = subjects.find((s) => s.masked === masked && s.mirrored === mirrored)!;
            const target = world.resource(viewTargetsKey).normal.get(subject.camera)!;
            const bytes = new DataView((await probeTexture(world, target.texture)).bytes);
            const packed = bytes.getUint32((16 * 33 + 16) * 4, true);
            const expected = [0, Math.SQRT1_2, mirrored ? Math.SQRT1_2 : -Math.SQRT1_2];
            for (let c = 0; c < 3; c++)
                expect(
                    Math.abs((((packed >>> (c * 10)) & 1023) / 1023) * 2 - 1 - expected[c]),
                ).toBeLessThan(1e-2);
        });
