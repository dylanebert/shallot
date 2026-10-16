import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { expectWarning } from "../../../scripts/test-console";
import { cube, Meshes, type MeshHandle, MeshInstance } from "../../core/mesh";
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
    Draws,
    MaterialPlugin,
    Materials,
    MaterialVertexInput,
    MeshMaterial,
    materialFragmentContext,
    materialLayout,
    materialType,
    materialVertexOutput,
    StandardMaterial,
    StandardRenderingPlugin,
} from "./index";
import { MeshRenderPlugin } from "./mesh-render";

setDefaultTimeout(TIMEOUT.node);

const parameters = d.struct({ unused: d.u32 });
const varyings = { tint: d.vec4f };
const streamLayout = materialLayout(parameters, {
    tint: { type: "attribute", element: d.vec4f },
});
const StreamVertexOutput = materialVertexOutput(varyings);
const StreamFragmentContext = materialFragmentContext(varyings);
const StreamMaterial = materialType({
    name: "StreamAttributeMaterial",
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
    world.resource(Meshes).add({
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
    const quad = world.resource(Meshes).add({
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

test("an attribute material and an unrelated mesh draw together without warning or an incompatible draw pair", async () => {
    const { world } = subjects()[1];
    const tinted = rebuild(world, [1, 0, 0, 1]);
    const unrelated = world.resource(Meshes).add({ name: "unrelated", vertices, indices });
    flushMeshes(world);
    for (const [x, mesh, material] of [
        [-0.7, tinted, world.resource(StreamMaterial).add({ unused: 0 })],
        [
            0.7,
            unrelated,
            world
                .resource(Materials)
                .add(StandardMaterial({ baseColor: [0, 1, 0, 1], unlit: true })),
        ],
    ] as const) {
        const eid = world.create();
        world.add(eid, Transform, { translation: [x, 0, 0], scale: [0.5, 0.5, 0.5] });
        world.add(eid, MeshInstance, { mesh });
        world.add(eid, MeshMaterial, material);
    }
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: 64, height: 32 });
    world.step(0);
    world.step(0);
    const frame = await captureTexture(world, camera);
    const pixel = (x: number) =>
        Array.from(frame.rgba.subarray((16 * 64 + x) * 4, (16 * 64 + x) * 4 + 4));
    expect(pixel(28)).toEqual([255, 0, 0, 255]);
    expect(pixel(36)).toEqual([0, 255, 0, 255]);
    expect(
        [...world.resource(Draws)].some(
            (draw) => draw.name.includes("StreamAttributeMaterial") && draw.mesh === unrelated,
        ),
    ).toBe(false);
});

for (const change of ["added-instance", "type", "mesh", "marked-raw-type"] as const)
    test(`${change} pairing changes warn once after draw publication`, () => {
        const { world } = subjects()[1];
        const missing = world.resource(Meshes).add({ name: change, vertices, indices });
        flushMeshes(world);
        world.step(0);
        const material = world.resource(StreamMaterial).add({ unused: 0 });
        const eid = world.create();
        world.add(eid, Transform);
        if (change === "mesh") {
            const valid = [...world.resource(Meshes).entries()].find(
                ([, mesh]) => mesh.name === "quad",
            )![0];
            world.add(eid, MeshInstance, { mesh: valid });
            world.add(eid, MeshMaterial, material);
        } else {
            world.add(eid, MeshInstance, { mesh: missing });
            world.add(eid, MeshMaterial, change === "added-instance" ? material : undefined);
        }
        if (change !== "added-instance") world.step(0);
        expectWarning(
            new RegExp(`mesh:material:StreamAttributeMaterial:${change}:\\d+.*attribute "tint"`),
        );
        if (change === "mesh") world.storage(MeshInstance).mesh.set(eid, missing);
        if (change === "type") world.storage(MeshMaterial).type.set(eid, material.type);
        if (change === "marked-raw-type") {
            const type = world.storage(MeshMaterial).type;
            type.column[eid] = material.type;
            type.markChanged(eid);
        }
        world.step(0);
        world.step(0);
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
        const handle = world.resource(Meshes).add({
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
