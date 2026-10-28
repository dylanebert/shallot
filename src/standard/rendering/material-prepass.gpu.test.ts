import { beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import { attachTexture, Camera, DirectionalLight, NormalPrepass } from "../../core/rendering";
import { Transform } from "../../core/transform";
import { lookAtRotation, probeTexture, type World } from "../../engine";
import {
    AlphaMode,
    type MaterialBinding,
    MaterialPlugin,
    type MaterialType,
    MeshMaterial,
    MeshRenderPlugin,
    materialLayout,
    materialType,
    StandardRenderer,
    StandardRenderingPlugin,
} from "./index";

setDefaultTimeout(TIMEOUT.gpu);
// Wider than any shadow atlas tile, so a write addressed by an atlas fragment's x lands past the camera's 8.
const WIDE = 4096;
const VARIANTS = ["plain", "varying"] as const;
type Variant = (typeof VARIANTS)[number];
const DEPTH_PASSES = ["prepasses", "shadow maps"] as const;
type DepthPass = (typeof DEPTH_PASSES)[number];
// Per variant: columns written by fragCoord.x, and the two pass texels of a material drawn in the
// color pass and one kind of depth pass.
const Written = {
    create: (world: World) => {
        const storage = (width: number) => {
            const texture = world.gpu.device.createTexture({
                size: [width, 1],
                format: "r32uint",
                usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
            });
            world.own(texture);
            return texture;
        };
        return {
            columns: { plain: storage(WIDE), varying: storage(WIDE) },
            passes: {
                plain: { prepasses: storage(2), "shadow maps": storage(2) },
                varying: { prepasses: storage(2), "shadow maps": storage(2) },
            },
        };
    },
};
const Unused = d.struct({ unused: d.f32 });
const columnsLayout = materialLayout(Unused, {
    columns: { type: "storage-texture", format: "r32uint", access: "write" },
});
const passesLayout = materialLayout(Unused, {
    passes: { type: "storage-texture", format: "r32uint", access: "write" },
});
// A varying routes a material through the varying color, prepass and shadow fragments.
const varying = {
    varyings: { tint: d.vec4f },
    vertex: (input: { world: d.v4f; worldNormal: d.v3f }) => {
        "use gpu";
        return { world: input.world, worldNormal: input.worldNormal, tint: d.vec4f(1) };
    },
};
// Marks the column of each fragment that writes, guarded to the color pass.
const guardedWrite = (variant: Variant) =>
    materialType({
        name: `GuardedStorageWrite_${variant}`,
        layout: columnsLayout,
        defaults: { unused: 0 },
        bindings: (world) => ({ columns: world.resource(Written).columns[variant] }),
        ...(variant === "varying" ? varying : {}),
        fragment: (ctx) => {
            "use gpu";
            if (!ctx.prepass) {
                const x = std.min(d.i32(ctx.fragCoord.x), WIDE - 1);
                std.textureStore(columnsLayout.$.columns, d.vec2i(x, 0), d.vec4u(1, 0, 0, 0));
            }
            return d.vec4f(1);
        },
    });
// Texel 0 marks a color-pass run, texel 1 a prepass or shadow run.
const passFlag = (variant: Variant, pass: DepthPass) =>
    materialType({
        name: `PrepassFlag_${variant}_${pass === "prepasses" ? "prepass" : "shadow"}`,
        layout: passesLayout,
        defaults: { unused: 0 },
        bindings: (world) => ({ passes: world.resource(Written).passes[variant][pass] }),
        ...(variant === "varying" ? varying : {}),
        depthPass: pass === "prepasses" ? { shadows: false } : { prepass: false },
        fragment: (ctx) => {
            "use gpu";
            const texel = d.vec2i(d.i32(ctx.prepass), 0);
            std.textureStore(passesLayout.$.passes, texel, d.vec4u(1, 0, 0, 0));
            return d.vec4f(1);
        },
    });
const materials: MaterialType<typeof Unused, Record<string, MaterialBinding>, any>[] = [
    ...VARIANTS.map(guardedWrite),
    ...VARIANTS.flatMap((variant) => DEPTH_PASSES.map((pass) => passFlag(variant, pass))),
];
function scene(world: World) {
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera, { antialias: 0 });
    world.add(camera, StandardRenderer);
    world.add(camera, NormalPrepass);
    attachTexture(world, camera, { width: 8, height: 4 });
    const sun = world.create();
    world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, -0.3, -0.3, -1) });
    world.add(sun, DirectionalLight, { illuminance: 10000 });
    world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 1);
    // one cell each of a 3x2 grid; masked rows run the material fragment in depth passes too
    materials.forEach((type, i) => {
        const mesh = world.create();
        world.add(mesh, Transform, {
            translation: [((i % 3) - 1) * 2.6, i < 3 ? 0.9 : -0.9, 0, 0],
            scale: [2.4, 1.6, 0.5, 0],
        });
        world.add(mesh, MeshInstance);
        world.add(
            mesh,
            MeshMaterial,
            world.resource(type).add({ unused: 0 }, { alphaMode: AlphaMode.Mask() }),
        );
    });
}
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            StandardRenderingPlugin,
            MeshRenderPlugin,
            ...materials.map((type) => MaterialPlugin(type)),
            {
                name: "PrepassScene",
                dependencies: [StandardRenderingPlugin, MeshRenderPlugin],
                initialize: scene,
            },
        ],
    },
]);
beforeAll(() => {
    const { world } = subjects()[0];
    world.step(1 / 60);
    world.step(1 / 60);
}, TIMEOUT.startup);

for (const variant of VARIANTS) {
    test(`a ${variant} material's storage write under !ctx.prepass lands only from the color pass`, async () => {
        const { world } = subjects()[0];
        const { bytes } = await probeTexture(world, world.resource(Written).columns[variant]);
        const columns = new Uint32Array(bytes, 0, WIDE);
        expect(columns.subarray(0, 8).some((written) => written === 1)).toBe(true);
        expect(columns.subarray(8).every((written) => written === 0)).toBe(true);
    });

    for (const pass of DEPTH_PASSES) {
        test(`a ${variant} material's ctx.prepass is false in the color pass and true in ${pass}`, async () => {
            const { world } = subjects()[0];
            const texture = world.resource(Written).passes[variant][pass];
            const { bytes } = await probeTexture(world, texture);
            expect(Array.from(new Uint32Array(bytes, 0, 2))).toEqual([1, 1]);
        });
    }
}
