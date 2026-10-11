import { expect, setDefaultTimeout, test } from "bun:test";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { expectWarning } from "../../../scripts/test-console";
import { CEILING } from "../../../scripts/test-tiers";
import { cube, type MeshHandle, MeshInstance, registerMesh } from "../../core/mesh";
import { clearMeshes, flushMeshes } from "../../core/mesh/mesh";
import {
    attachTexture,
    Camera,
    captureTexture,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { StandardRenderer } from "./forward";
import {
    MaterialPlugin,
    MaterialVertexInput,
    MeshMaterial,
    materialFragmentContext,
    materialLayout,
    materialType,
    materialVertexOutput,
    StandardRenderingPlugin,
} from "./index";
import { MeshRenderPlugin } from "./mesh-render";

setDefaultTimeout(CEILING.node);

const parameters = d.struct({ unused: d.u32 });
const varyings = { tint: d.vec4f };
const streamLayout = materialLayout(parameters, {
    tint: { type: "attribute", element: d.vec4f },
});
const StreamVertexOutput = materialVertexOutput(varyings);
const StreamFragmentContext = materialFragmentContext(varyings);
const StreamMaterial = materialType({
    name: "StreamAttributeMaterial",
    parameters,
    layout: streamLayout,
    varyings,
    vertex: tgpu.fn(
        [MaterialVertexInput],
        StreamVertexOutput,
    )((input) => {
        "use gpu";
        return StreamVertexOutput({
            world: input.world,
            worldNormal: input.worldNormal,
            tint: streamLayout.$.tint[input.vidx],
        });
    }),
    fragment: tgpu.fn(
        [StreamFragmentContext],
        d.vec4f,
    )((ctx) => {
        "use gpu";
        return d.vec4f(ctx.tint);
    }),
    defaults: { unused: 0 },
});
const streamPlugin = MaterialPlugin(StreamMaterial);
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [StandardRenderingPlugin, MeshRenderPlugin, streamPlugin] },
]);
const vertices = new Float32Array([
    -1, -1, 0, 0, 0, 0, 1, 0, 1, -1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, -1, 1, 0, 0, 0, 0, 1,
    1,
]);
const indices = new Uint32Array([0, 1, 2, 0, 2, 3]);
function initAttributeMeshes(world: ReturnType<typeof subjects>[number]["world"]): void {
    clearMeshes(world);
    const geometry = cube();
    registerMesh(world, {
        name: "cube",
        ...geometry,
        attributes: {
            tint: {
                element: d.vec4f,
                data: new Float32Array((geometry.vertices.length / 8) * 4).fill(1),
            },
        },
    });
}

function rebuild(
    world: ReturnType<typeof subjects>[number]["world"],
    colour: number[],
): MeshHandle {
    initAttributeMeshes(world);
    const quad = registerMesh(world, {
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
    flushMeshes(world);
    return quad;
}

test("a material type reads a quad's vec4f stream and rebinds after clear and rebuild", async () => {
    const { world } = subjects()[0];
    const quad = rebuild(world, [1, 0, 0, 1]);
    const material = world.resource(StreamMaterial).add({ unused: 0 });
    const eid = world.create();
    world.add(eid, Transform);
    world.add(eid, MeshInstance, { mesh: quad });
    world.add(eid, MeshMaterial, material);
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: 8, height: 8 });
    world.step(0);
    expect(
        Array.from((await captureTexture(world, camera)).rgba.subarray(4 * 36, 4 * 36 + 4)),
    ).toEqual([255, 0, 0, 255]);
    const rebuiltQuad = rebuild(world, [0, 1, 0, 1]);
    world.storage(MeshInstance).mesh.set(eid, rebuiltQuad);
    world.step(0);
    expect(
        Array.from((await captureTexture(world, camera)).rgba.subarray(4 * 36, 4 * 36 + 4)),
    ).toEqual([0, 255, 0, 255]);
});

for (const mesh of ["missing", "wrong"])
    test(`a ${mesh} stream refusal names mesh, material type and attribute and warns once`, () => {
        expectWarning(
            new RegExp(
                `mesh:material:StreamAttributeMaterial:${mesh}:\\d+\\" skipped — mesh \\"${mesh}\\".*attribute \\"tint\\"`,
            ),
        );
        const { world } = subjects()[0];
        initAttributeMeshes(world);
        const handle = registerMesh(world, {
            name: mesh,
            vertices,
            indices,
            ...(mesh === "wrong"
                ? { attributes: { tint: { element: d.f32, data: new Float32Array(4) } } }
                : {}),
        });
        flushMeshes(world);
        const eid = world.create();
        world.add(eid, Transform);
        world.add(eid, MeshInstance, { mesh: handle });
        world.add(eid, MeshMaterial, world.resource(StreamMaterial).add({ unused: 0 }));
        world.step(0);
        world.step(0);
    });
