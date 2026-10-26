import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import { attachTexture, Camera, NormalPrepass } from "../../core/rendering";
import { viewTargetsKey } from "../../core/rendering/targets";
import { Transform } from "../../core/transform";
import { probeTexture, type World } from "../../engine";
import { StandardRenderer } from "./forward";
import {
    AlphaMode,
    applyPbrLighting,
    extendedMaterial,
    lit,
    type MaterialHandle,
    MaterialPlugin,
    Materials,
    MeshMaterial,
    materialType,
    pbrInputFromStandardMaterial,
    StandardMaterial,
    StandardRenderingPlugin,
} from "./index";
import { MeshRenderPlugin } from "./mesh-render";

setDefaultTimeout(TIMEOUT.gpu);
// Each material lights its +Z face with normalize(n + y), which is (0, √½, √½) there.
const bent = [0, Math.SQRT1_2, Math.SQRT1_2];
const BentParameters = d.struct({ color: d.vec4f });
const BentLit = materialType({
    name: "BentLit",
    parameters: BentParameters,
    defaults: { color: d.vec4f(1) },
    depthPass: { prepass: true },
    fragment: (ctx) => {
        "use gpu";
        const normal = std.normalize(std.add(ctx.worldNormal, d.vec3f(0, 1, 0)));
        return d.vec4f(lit(ctx.parameters.color.xyz, normal), ctx.parameters.color.w);
    },
});
const BentVarying = materialType({
    name: "BentVarying",
    parameters: BentParameters,
    defaults: { color: d.vec4f(1) },
    depthPass: { prepass: true },
    varyings: { tint: d.vec4f },
    vertex: (input) => {
        "use gpu";
        return { world: input.world, worldNormal: input.worldNormal, tint: input.parameters.color };
    },
    fragment: (ctx) => {
        "use gpu";
        const normal = std.normalize(std.add(ctx.worldNormal, d.vec3f(0, 1, 0)));
        return d.vec4f(lit(ctx.tint.xyz, normal), ctx.tint.w);
    },
});
const BentStandard = extendedMaterial(Materials, {
    name: "BentStandard",
    fragment: (ctx) => {
        "use gpu";
        const input = pbrInputFromStandardMaterial(ctx);
        input.worldNormal = std.normalize(std.add(ctx.worldNormal, d.vec3f(0, 1, 0)));
        return applyPbrLighting(input);
    },
});
const cameras = new Map<string, number>();
function scene(world: World) {
    const subjects: [string, () => MaterialHandle][] = [
        ["lit Opaque", () => world.resource(BentLit).add({ color: d.vec4f(1) })],
        [
            "lit Mask",
            () =>
                world
                    .resource(BentLit)
                    .add({ color: d.vec4f(1) }, { alphaMode: AlphaMode.Mask(0.5) }),
        ],
        ["one-varying Opaque", () => world.resource(BentVarying).add({ color: d.vec4f(1) })],
        ["extendedMaterial Opaque", () => world.resource(BentStandard).add(StandardMaterial())],
    ];
    for (let i = 0; i < subjects.length; i++) {
        const [label, material] = subjects[i]!;
        const eid = world.create();
        world.add(eid, Transform, { translation: [i * 10, 0, 0, 0] });
        world.add(eid, MeshInstance);
        world.add(eid, MeshMaterial, material());
        const camera = world.create();
        world.add(camera, Transform, { translation: [i * 10, 0, 4, 0] });
        world.add(camera, Camera, { antialias: 0 });
        world.add(camera, StandardRenderer);
        world.add(camera, NormalPrepass);
        attachTexture(world, camera, { width: 33, height: 33 });
        cameras.set(label, camera);
    }
}
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            StandardRenderingPlugin,
            MeshRenderPlugin,
            MaterialPlugin(BentLit),
            MaterialPlugin(BentVarying),
            MaterialPlugin(BentStandard),
            {
                name: "BentNormalScene",
                dependencies: [StandardRenderingPlugin, MeshRenderPlugin],
                initialize: scene,
            },
        ],
    },
]);
for (const label of ["lit Opaque", "lit Mask", "one-varying Opaque", "extendedMaterial Opaque"])
    test(`the ${label} material's prepass normal is the normal its fragment lit with`, async () => {
        const { world } = subjects()[0];
        world.step(1 / 60);
        const target = world.resource(viewTargetsKey).normal.get(cameras.get(label)!)!;
        const bytes = new DataView((await probeTexture(world, target.texture)).bytes);
        const packed = bytes.getUint32((16 * 33 + 16) * 4, true);
        for (let c = 0; c < 3; c++) {
            const normal = (((packed >>> (c * 10)) & 1023) / 1023) * 2 - 1;
            expect(Math.abs(normal - bent[c]!)).toBeLessThan(1e-2);
        }
    });
