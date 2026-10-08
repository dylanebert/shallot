import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { registerMesh } from "../../core/mesh";
import { clearMeshes, flushMeshes } from "../../core/mesh/mesh";
import { initMeshes } from "../../core/mesh/primitives";
import {
    attachTexture,
    Camera,
    captureTexture,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { fsCtxSchema, registerSurface, surfaceLayout, VsIn, vsPatchSchema } from "./contract";
import { StandardRenderer } from "./forward";
import { StandardRenderingPlugin } from "./index";
import { MeshRenderPlugin } from "./mesh-render";
import { DrawIndexedIndirect, Draws } from "./registry";

setDefaultTimeout(CEILING.gpu);
function makeSurface() {
    const layout = surfaceLayout({ tint: { type: "attribute", element: d.vec4f } });
    const varyings = { tint: d.vec4f };
    return {
        name: "stream",
        layout,
        varyings,
        screen: true,
        vs: tgpu.fn(
            [VsIn],
            vsPatchSchema(varyings),
        )((v) => {
            "use gpu";
            return {
                world: v.world,
                worldNormal: v.worldNormal,
                clip: d.vec4f(v.localPos.x, v.localPos.y, 0.5, 1),
                tint: layout.$.tint[v.vidx],
            };
        }),
        fs: tgpu.fn(
            [fsCtxSchema(varyings)],
            d.vec4f,
        )((v) => {
            "use gpu";
            return d.vec4f(v.tint);
        }),
    };
}
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [StandardRenderingPlugin, MeshRenderPlugin] },
]);
const vertices = new Float32Array([
    -1, -1, 0, 0, 0, 0, 1, 0, 1, -1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, -1, 1, 0, 0, 0, 0, 1,
    1,
]);
const indices = new Uint32Array([0, 1, 2, 0, 2, 3]);
function rebuild(world: ReturnType<typeof subjects>[number]["world"], colour: number[]) {
    clearMeshes(world);
    initMeshes(world);
    registerMesh(world, {
        name: "quad",
        vertices,
        indices,
        attributes: {
            tint: {
                element: d.vec4f,
                data: new Float32Array(Array.from({ length: 4 }, () => colour).flat()),
            },
        },
    });
    registerMesh(world, {
        name: "wrong",
        vertices,
        indices,
        attributes: { tint: { element: d.f32, data: new Float32Array(4) } },
    });
    flushMeshes(world);
}

test("a custom surface reads a quad's vec4f stream and rebinds after clear and rebuild", async () => {
    const { world } = subjects()[0];
    rebuild(world, [1, 0, 0, 1]);
    registerSurface(world, makeSurface());
    const camera = world.create();
    world.add(camera, Transform);
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: 8, height: 8 });
    const indirect = world.gpu.root
        .createBuffer(DrawIndexedIndirect, {
            indexCount: 6,
            instanceCount: 1,
            firstIndex: 0,
            baseVertex: 0,
            firstInstance: 0,
        })
        .$usage("indirect");
    world
        .resource(Draws)
        .register({ name: "quad-draw", surface: "stream", mesh: "quad", args: { indirect } });
    world.step(0);
    expect(
        Array.from((await captureTexture(world, camera)).rgba.subarray(4 * 36, 4 * 36 + 4)),
    ).toEqual([255, 0, 0, 255]);
    rebuild(world, [0, 1, 0, 1]);
    world.step(0);
    expect(
        Array.from((await captureTexture(world, camera)).rgba.subarray(4 * 36, 4 * 36 + 4)),
    ).toEqual([0, 255, 0, 255]);
});

for (const mesh of ["cube", "wrong"])
    test(`a ${mesh} stream refusal names mesh, surface and attribute and warns once`, () => {
        const { world } = subjects()[0];
        registerSurface(world, makeSurface());
        flushMeshes(world);
        const indirect = world.gpu.root
            .createBuffer(DrawIndexedIndirect, {
                indexCount: 6,
                instanceCount: 1,
                firstIndex: 0,
                baseVertex: 0,
                firstInstance: 0,
            })
            .$usage("indirect");
        world
            .resource(Draws)
            .register({ name: `${mesh}-draw`, surface: "stream", mesh, args: { indirect } });
        const warn = spyOn(console, "warn").mockImplementation(() => {});
        try {
            world.step(0);
            world.step(0);
            const messages = warn.mock.calls
                .map((call) => call.join(" "))
                .filter((message) => message.includes(`${mesh}-draw`));
            expect(messages).toHaveLength(1);
            expect(messages[0]).toContain(`mesh "${mesh}"`);
            expect(messages[0]).toContain('surface "stream"');
            expect(messages[0]).toContain('attribute "tint"');
        } finally {
            warn.mockRestore();
        }
    });
