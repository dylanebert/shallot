import { beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { Meshes, MeshInstance } from "../../core/mesh";
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
const VARIANTS = ["plain", "varying"] as const;
type Variant = (typeof VARIANTS)[number];
const DEPTH_PASSES = ["prepasses", "shadow maps"] as const;
type DepthPass = (typeof DEPTH_PASSES)[number];
// Rows by tag: 0 draws unmirrored, 1 mirrored, 2 nothing. A fragment writes two texels at
// ((prepass * 3 + tag) * 2): 1 + frontFacing, then material + 1.
const TAGS = 3;
const WIDTH = 2 * TAGS * 2;
const Seen = {
    create: (world: World) => {
        const storage = () => {
            const texture = world.gpu.device.createTexture({
                size: [WIDTH, 1],
                format: "r32uint",
                usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
            });
            world.own(texture);
            return texture;
        };
        return {
            plain: { prepasses: storage(), "shadow maps": storage() },
            varying: { prepasses: storage(), "shadow maps": storage() },
        };
    },
};
const Tag = d.struct({ tag: d.u32 });
const seenLayout = materialLayout(Tag, {
    seen: { type: "storage-texture", format: "r32uint", access: "write" },
});
// A varying routes a material through the varying color, prepass and shadow stages.
const varying = {
    varyings: { tint: d.vec4f },
    vertex: (input: { world: d.v4f; worldNormal: d.v3f }) => {
        "use gpu";
        return { world: input.world, worldNormal: input.worldNormal, tint: d.vec4f(1) };
    },
};
const recorder = (variant: Variant, pass: DepthPass) =>
    materialType({
        name: `MirrorRecorder_${variant}_${pass === "prepasses" ? "prepass" : "shadow"}`,
        layout: seenLayout,
        defaults: { tag: 2 },
        bindings: (world) => ({ seen: world.resource(Seen)[variant][pass] }),
        ...(variant === "varying" ? varying : {}),
        depthPass: pass === "prepasses" ? { shadows: false } : { prepass: false },
        fragment: (ctx) => {
            "use gpu";
            const at = (d.i32(ctx.prepass) * TAGS + d.i32(std.min(ctx.parameters.tag, 2))) * 2;
            std.textureStore(
                seenLayout.$.seen,
                d.vec2i(at, 0),
                d.vec4u(1 + d.u32(ctx.frontFacing), 0, 0, 0),
            );
            std.textureStore(
                seenLayout.$.seen,
                d.vec2i(at + 1, 0),
                d.vec4u(ctx.material + 1, 0, 0, 0),
            );
            return d.vec4f(1);
        },
    });
const recorders: MaterialType<typeof Tag, Record<string, MaterialBinding>, any>[] =
    VARIANTS.flatMap((variant) => DEPTH_PASSES.map((pass) => recorder(variant, pass)));
const rows: { unmirrored: number; mirrored: number }[] = [];
function scene(world: World) {
    // A one-sided quad whose front face, counter-clockwise, looks along +Z toward the camera and sun.
    const mesh = world.resource(Meshes).add({
        name: "mirror-quad",
        vertices: new Float32Array([
            -1, -1, 0, 0, 0, 0, 1, 0, 1, -1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, -1, 1, 0, 0,
            0, 0, 1, 1,
        ]),
        indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    });
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 8] });
    world.add(camera, Camera, { antialias: 0 });
    world.add(camera, StandardRenderer);
    world.add(camera, NormalPrepass);
    attachTexture(world, camera, { width: 16, height: 8 });
    const sun = world.create();
    world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, -0.2, -0.2, -1) });
    world.add(sun, DirectionalLight, { illuminance: 10000 });
    world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 1);
    recorders.forEach((type, i) => {
        const materials = world.resource(type);
        const options = { alphaMode: AlphaMode.Mask(), cullMode: "none" } as const;
        const unmirrored = materials.add({ tag: 0 }, options);
        const mirrored = materials.add({ tag: 1 }, options);
        materials.add({ tag: 2 }, options);
        rows.push({ unmirrored: unmirrored.material, mirrored: mirrored.material });
        for (const [column, handle, sx] of [
            [0, unmirrored, 1.1],
            [1, mirrored, -1.1],
        ] as const) {
            const quad = world.create();
            world.add(quad, Transform, {
                translation: [(i - 1.5) * 2.5 + (column - 0.5) * 1.2, 0, 0],
                scale: [sx * 0.5, 0.8, 1],
            });
            world.add(quad, MeshInstance, { mesh });
            world.add(quad, MeshMaterial, handle);
        }
    });
}
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            StandardRenderingPlugin,
            MeshRenderPlugin,
            ...recorders.map((type) => MaterialPlugin(type)),
            {
                name: "MirrorScene",
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

const TRUE = 2;
VARIANTS.forEach((variant, v) => {
    DEPTH_PASSES.forEach((pass, p) => {
        test(`a ${variant} material on a mirrored instance reads its row and logical front face in the color pass and ${pass}`, async () => {
            const { world } = subjects()[0];
            const { bytes } = await probeTexture(world, world.resource(Seen)[variant][pass]);
            const seen = Array.from(new Uint32Array(bytes, 0, WIDTH));
            const row = rows[v * DEPTH_PASSES.length + p]!;
            const slot = (prepass: number, tag: number) =>
                seen.slice((prepass * TAGS + tag) * 2, (prepass * TAGS + tag) * 2 + 2);
            for (const prepass of [0, 1]) {
                expect({ prepass, unmirrored: slot(prepass, 0) }).toEqual({
                    prepass,
                    unmirrored: [TRUE, row.unmirrored + 1],
                });
                expect({ prepass, mirrored: slot(prepass, 1) }).toEqual({
                    prepass,
                    mirrored: [TRUE, row.mirrored + 1],
                });
                expect({ prepass, spare: slot(prepass, 2) }).toEqual({ prepass, spare: [0, 0] });
            }
        });
    });
});
