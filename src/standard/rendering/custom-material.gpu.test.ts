import { expect, setDefaultTimeout, test } from "bun:test";
import tgpu, { readFromArrayBuffer } from "typegpu";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    captureTexture,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { probeBuffer } from "../../engine/runtime";
import {
    Draws,
    MaterialPlugin,
    MeshMaterial,
    MeshRenderPlugin,
    materialFragmentContext,
    materialLayout,
    materialType,
    materialTypeId,
    StandardRenderer,
    StandardRenderingPlugin,
} from "./index";

setDefaultTimeout(CEILING.gpu);

const TintParameters = d.struct({ tint: d.vec4f });
const tintLayout = materialLayout(TintParameters, {});
const TintContext = materialFragmentContext();
const TintMaterial = materialType({
    name: "PublicTwoTintMaterial",
    parameters: TintParameters,
    layout: tintLayout,
    fragment: tgpu.fn(
        [TintContext],
        d.vec4f,
    )((ctx) => {
        "use gpu";
        return TintParameters(tintLayout.$.materialParameters[ctx.material]).tint;
    }),
    defaults: { tint: d.vec4f(1) },
});
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [StandardRenderingPlugin, MeshRenderPlugin, MaterialPlugin(TintMaterial)],
    },
]);

test("the public material type renders two typed tint rows together and updates the red row next frame", async () => {
    const { world } = subjects()[0];
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: 64, height: 32 });

    const materials = world.resource(TintMaterial);
    const red = materials.add({ tint: d.vec4f(1, 0, 0, 1) });
    const green = materials.add({ tint: d.vec4f(0, 1, 0, 1) });
    const entities = [
        { x: -0.7, material: red },
        { x: 0.7, material: green },
    ];
    for (const { x, material } of entities) {
        const eid = world.create();
        world.add(eid, Transform, { translation: [x, 0, 0, 0], scale: [0.5, 0.5, 0.5, 0] });
        world.add(eid, MeshInstance);
        world.add(eid, MeshMaterial, material);
    }

    const sample = async () => {
        world.step(0);
        world.step(0);
        const frame = await captureTexture(world, camera);
        const pixel = (x: number, y = 16) =>
            Array.from(frame.rgba.subarray((y * 64 + x) * 4, (y * 64 + x) * 4 + 4));
        return { left: pixel(28), right: pixel(36) };
    };
    expect(await sample()).toEqual({ left: [255, 0, 0, 255], right: [0, 255, 0, 255] });
    const redRow = await probeBuffer(world, materials.table.buffer, {
        offset: red.material * d.sizeOf(TintParameters),
        size: d.sizeOf(TintParameters),
    });
    expect(readFromArrayBuffer(redRow.bytes, TintParameters).tint).toEqual(d.vec4f(1, 0, 0, 1));

    const type = materialTypeId(world, TintMaterial);
    const draw = [...world.resource(Draws)].find((candidate) => candidate.materialType === type)!;
    const args = await probeBuffer(world, world.gpu.root.unwrap(draw.args.indirect), {
        offset: draw.args.offset ?? 0,
        size: 20,
    });
    expect(new Uint32Array(args.bytes)[1]).toBe(2);

    materials.update(red, { tint: d.vec4f(0, 0, 1, 1) });
    expect(await sample()).toEqual({ left: [0, 0, 255, 255], right: [0, 255, 0, 255] });
});
