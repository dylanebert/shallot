import { afterAll, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import {
    compileGpuFile,
    createTestGpuApp,
    disposeGpuApps,
    gpuScene,
} from "../../../scripts/gpu.fixture";
import { expectWarning } from "../../../scripts/test-console";
import { Meshes, type MeshHandle, MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    captureTexture,
    DepthPrepass,
    DirectionalLight,
    Tonemapping,
    TonemappingMethod,
    Views,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { lookAtRotation, type Plugin, precompile, type World } from "../../engine";
import { probeBuffer } from "../../engine/runtime";
import { DEFAULT_PLUGINS } from "../../standard";
import { sunShadowView } from "./atlas";
import {
    AlphaMode,
    type MaterialHandle,
    MaterialPlugin,
    Materials,
    MeshMaterial,
    materialType,
    registerProducerTable,
    StandardMaterial,
    StandardRenderer,
} from "./index";
import { MeshDrawBuffers } from "./preprocess";
import { Draws } from "./registry";

setDefaultTimeout(TIMEOUT.node);

const SIDE = 16;
const CENTER = ((SIDE / 2) * SIDE + SIDE / 2) * 4;
const BLUE = [0, 0, 255, 255];
const RED = [255, 0, 0, 255];
const GREEN = [0, 255, 0, 255];

const ColorParameters = d.struct({ color: d.vec4f });
const Plain = materialType({
    name: "CullPlain",
    parameters: ColorParameters,
    defaults: { color: d.vec4f(0, 0, 1, 1) },
    depthPass: { prepass: true, shadows: true },
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(ctx.parameters.color);
    },
});
const Facing = materialType({
    name: "CullFacing",
    parameters: ColorParameters,
    defaults: { color: d.vec4f(1) },
    fragment: (ctx) => {
        "use gpu";
        if (ctx.frontFacing) return d.vec4f(1, 0, 0, 1);
        return d.vec4f(0, 1, 0, 1);
    },
});
const FacingPlugin = MaterialPlugin(Facing);
const cullPlugins: Plugin[] = [...DEFAULT_PLUGINS, MaterialPlugin(Plain)];

/** A one-sided 2x2 quad at z = 0 whose front face, counter-clockwise, looks along +Z. */
function quad(world: World, name: string): MeshHandle {
    return world.resource(Meshes).add({
        name,
        vertices: new Float32Array([
            -1, -1, 0, 0, 0, 0, 1, 0, 1, -1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, -1, 1, 0, 0,
            0, 0, 1, 1,
        ]),
        indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    });
}

/** A camera on the z axis at `z`, looking at the origin. */
function camera(world: World, z: number, depth = false): number {
    const eid = world.create();
    world.add(eid, Transform, {
        translation: [0, 0, z, 0],
        rotation: z < 0 ? lookAtRotation(0, 0, z, 0, 0, 0) : [0, 0, 0, 1],
    });
    world.add(eid, Camera, { antialias: 0 });
    world.add(eid, StandardRenderer);
    world.add(eid, Tonemapping, { method: TonemappingMethod.None });
    if (depth) world.add(eid, DepthPrepass);
    attachTexture(world, eid, { width: SIDE, height: SIDE });
    return eid;
}

async function pixels(world: World, eid: number) {
    const rgba = (await captureTexture(world, eid)).rgba;
    return {
        center: Array.from(rgba.subarray(CENTER, CENTER + 4)),
        clear: Array.from(rgba.subarray(0, 4)),
    };
}

const faces = gpuScene(import.meta.path, { defaults: false, plugins: cullPlugins }, (world) => {
    const mesh = quad(world, "cull-faces-quad");
    const subject = world.create();
    world.add(subject, Transform);
    world.add(subject, MeshInstance, { mesh });
    const plain = world.resource(Plain);
    const rows = {
        back: plain.add({ color: d.vec4f(0, 0, 1, 1) }),
        none: plain.add({ color: d.vec4f(0, 0, 1, 1) }, { cullMode: "none" }),
        front: plain.add({ color: d.vec4f(0, 0, 1, 1) }, { cullMode: "front" }),
        blend: plain.add(
            { color: d.vec4f(0, 0, 1, 0.5) },
            { alphaMode: AlphaMode.Blend, cullMode: "none" },
        ),
    };
    world.add(subject, MeshMaterial, rows.back);
    return { subject, rows, front: camera(world, 5), back: camera(world, -5) };
});

async function views(row: keyof ReturnType<typeof faces>["scene"]["rows"]) {
    const { world, scene } = faces();
    world.add(scene.subject, MeshMaterial, scene.rows[row]);
    world.step(0);
    return { front: await pixels(world, scene.front), back: await pixels(world, scene.back) };
}

test("cullMode back, none and front choose which faces of a row draw", async () => {
    const back = await views("back");
    expect(back.front.center).toEqual(BLUE);
    expect(back.back.center).toEqual(back.back.clear);
    const none = await views("none");
    expect(none.front.center).toEqual(BLUE);
    expect(none.back.center).toEqual(BLUE);
    const front = await views("front");
    expect(front.front.center).toEqual(front.front.clear);
    expect(front.back.center).toEqual(BLUE);
});

test("a blended row with cullMode none draws its back face", async () => {
    const blend = await views("blend");
    expect(blend.back.center).not.toEqual(blend.back.clear);
    expect(blend.back.center).toEqual(blend.front.center);
});

const facingApps: Awaited<ReturnType<typeof createTestGpuApp>>[] = [];
afterAll(() => disposeGpuApps(facingApps));
// Built apart so a fragment that cannot read frontFacing fails its own claim.
const facing = compileGpuFile(import.meta.path, async () => {
    let scene!: {
        subject: number;
        rows: Record<"back" | "none" | "front", MaterialHandle>;
        front: number;
        back: number;
    };
    try {
        const app = await createTestGpuApp({
            defaults: false,
            plugins: [
                ...DEFAULT_PLUGINS,
                FacingPlugin,
                {
                    name: "CullFacingScene",
                    dependencies: [...DEFAULT_PLUGINS, FacingPlugin],
                    initialize(world) {
                        const mesh = quad(world, "cull-facing-quad");
                        const subject = world.create();
                        world.add(subject, Transform);
                        world.add(subject, MeshInstance, { mesh });
                        const materials = world.resource(Facing);
                        const rows = {
                            back: materials.add({ color: d.vec4f(1) }),
                            none: materials.add({ color: d.vec4f(1) }, { cullMode: "none" }),
                            front: materials.add({ color: d.vec4f(1) }, { cullMode: "front" }),
                        };
                        world.add(subject, MeshMaterial, rows.back);
                        scene = { subject, rows, front: camera(world, 5), back: camera(world, -5) };
                    },
                },
            ],
        });
        facingApps.push(app);
        return { world: app.world, scene };
    } catch (error) {
        return error as Error;
    }
});

test("ctx.frontFacing tells front from back fragments in every cull mode", async () => {
    const built = facing();
    if (built instanceof Error) throw built;
    const { world, scene } = built;
    const seen = async (row: "back" | "none" | "front") => {
        world.add(scene.subject, MeshMaterial, scene.rows[row]);
        world.step(0);
        return {
            front: (await pixels(world, scene.front)).center,
            back: (await pixels(world, scene.back)).center,
        };
    };
    expect((await seen("back")).front).toEqual(RED);
    const none = await seen("none");
    expect(none.front).toEqual(RED);
    expect(none.back).toEqual(GREEN);
    expect((await seen("front")).back).toEqual(GREEN);
});

test("ctx.frontFacing names a mirrored instance's logical face in every cull mode", async () => {
    const built = facing();
    if (built instanceof Error) throw built;
    const { world, scene } = built;
    // Mirrored, the quad's front face winds clockwise toward the front camera.
    world.add(scene.subject, Transform, { scale: [-1, 1, 1] });
    try {
        const seen = async (row: "back" | "none" | "front") => {
            world.add(scene.subject, MeshMaterial, scene.rows[row]);
            world.step(0);
            return {
                front: await pixels(world, scene.front),
                back: await pixels(world, scene.back),
            };
        };
        const back = await seen("back");
        expect(back.front.center).toEqual(back.front.clear);
        expect(back.back.center).toEqual(GREEN);
        const none = await seen("none");
        expect(none.front.center).toEqual(RED);
        expect(none.back.center).toEqual(GREEN);
        const front = await seen("front");
        expect(front.front.center).toEqual(RED);
        expect(front.back.center).toEqual(front.back.clear);
    } finally {
        world.add(scene.subject, Transform);
    }
});

const lit = gpuScene(import.meta.path, { defaults: false, plugins: cullPlugins }, (world) => {
    const mesh = quad(world, "cull-lit-quad");
    const subject = world.create();
    world.add(subject, Transform);
    world.add(subject, MeshInstance, { mesh });
    const materials = world.resource(Materials);
    const values = { baseColor: [0.8, 0.6, 0.4, 1] as const, perceptualRoughness: 0.6 };
    const rows = {
        doubleSided: materials.add(StandardMaterial({ ...values, doubleSided: true }), {
            cullMode: "none",
        }),
        singleSided: materials.add(StandardMaterial(values), { cullMode: "none" }),
    };
    world.add(subject, MeshMaterial, rows.doubleSided);
    const sun = world.create();
    world.add(sun, Transform);
    world.add(sun, DirectionalLight, { illuminance: 1500 });
    return { subject, rows, sun, front: camera(world, 5), back: camera(world, -5) };
});

test("a doubleSided StandardMaterial shades its back face with the flipped normal", async () => {
    const { world, scene } = lit();
    const capture = async (row: "doubleSided" | "singleSided", toward: number, eid: number) => {
        world.add(scene.subject, MeshMaterial, scene.rows[row]);
        // The sun travels along `toward` on z, lighting the face that looks back along it.
        world.add(scene.sun, Transform, { rotation: lookAtRotation(0, 0, 0, 0, 0, toward) });
        world.step(0);
        return (await pixels(world, eid)).center;
    };
    const frontLit = await capture("doubleSided", -1, scene.front);
    const backLit = await capture("doubleSided", 1, scene.back);
    const backUnflipped = await capture("singleSided", 1, scene.back);
    console.log(
        `double-sided: front ${frontLit.join(",")}, mirrored back ${backLit.join(",")}, single-sided back ${backUnflipped.join(",")}`,
    );
    expect(
        Math.max(...frontLit.map((channel, i) => Math.abs(channel - backLit[i]!))),
    ).toBeLessThanOrEqual(1);
    expect(backUnflipped[0]!).toBeLessThan(backLit[0]! - 20);
});

test("a mirrored doubleSided StandardMaterial shades each face as the unmirrored one", async () => {
    const { world, scene } = lit();
    world.add(scene.subject, MeshMaterial, scene.rows.doubleSided);
    const capture = async (scale: number, toward: number, eid: number) => {
        world.add(scene.subject, Transform, { scale: [scale, 1, 1] });
        world.add(scene.sun, Transform, { rotation: lookAtRotation(0, 0, 0, 0, 0, toward) });
        world.step(0);
        return (await pixels(world, eid)).center;
    };
    try {
        for (const [toward, eid] of [
            [-1, scene.front],
            [1, scene.back],
        ] as const) {
            const unmirrored = await capture(1, toward, eid);
            const mirrored = await capture(-1, toward, eid);
            expect({ toward, mirrored }).toEqual({ toward, mirrored: unmirrored });
        }
    } finally {
        world.add(scene.subject, Transform);
    }
});

const depthLayout = tgpu.bindGroupLayout({
    front: { texture: d.textureDepth2d() },
    back: { texture: d.textureDepth2d() },
    sun: { texture: d.textureDepth2d() },
    out: { storage: d.arrayOf(d.f32), access: "mutable" },
});
// TypeGPU's textureDimensions overloads do not include depth textures. out holds both view
// centres, then the count of sun-atlas texels with any depth.
const depthSample = tgpu
    .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [1] })(/* wgsl */ `{
    bound.out[0] = textureLoad(bound.front, vec2i(${SIDE / 2}, ${SIDE / 2}), 0);
    bound.out[1] = textureLoad(bound.back, vec2i(${SIDE / 2}, ${SIDE / 2}), 0);
    let size = textureDimensions(bound.sun);
    var covered = 0u;
    for (var y = 0u; y < size.y; y += 4u) {
        for (var x = 0u; x < size.x; x += 4u) {
            if (textureLoad(bound.sun, vec2i(i32(x), i32(y)), 0) > 0.0) { covered += 1u; }
        }
    }
    bound.out[2] = f32(covered);
}`)
    .$uses({ bound: depthLayout.$ });
const depthProbe = {
    create(world: World) {
        const pipeline = world.gpu.root
            .createComputePipeline({ compute: depthSample })
            .$name("material-cull-depth-probe");
        precompile(world, "material-cull-depth-probe", () => [pipeline]);
        return world.gpu.root.unwrap(pipeline);
    },
};

const depth = gpuScene(
    import.meta.path,
    {
        defaults: false,
        plugins: [
            ...cullPlugins,
            {
                name: "CullDepthProbe",
                warm(world) {
                    world.resource(depthProbe);
                },
            },
        ],
    },
    (world) => {
        const mesh = quad(world, "cull-depth-quad");
        const subject = world.create();
        world.add(subject, Transform);
        world.add(subject, MeshInstance, { mesh });
        const plain = world.resource(Plain);
        const rows = {
            back: plain.add({ color: d.vec4f(1) }),
            none: plain.add({ color: d.vec4f(1) }, { cullMode: "none" }),
        };
        world.add(subject, MeshMaterial, rows.back);
        // The sun travels along +Z, lighting the quad's back face.
        const sun = world.create();
        world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, 0.1, 0.2, 1) });
        world.add(sun, DirectionalLight, { shadowMapsEnabled: 1 });
        const out = world.gpu.device.createBuffer({
            size: 12,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        world.own(out);
        return { subject, rows, out, front: camera(world, 5, true), back: camera(world, -5, true) };
    },
);

async function depths(row: "back" | "none") {
    const { world, scene } = depth();
    world.add(scene.subject, MeshMaterial, scene.rows[row]);
    const device = world.gpu.device;
    device.pushErrorScope("validation");
    world.step(0);
    const group = world.gpu.root.createBindGroup(depthLayout, {
        front: world.resource(Views).get(scene.front)!.depth!,
        back: world.resource(Views).get(scene.back)!.depth!,
        sun: sunShadowView(world)!,
        out: scene.out,
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(world.resource(depthProbe));
    pass.setBindGroup(0, world.gpu.root.unwrap(group));
    pass.dispatchWorkgroups(1);
    pass.end();
    device.queue.submit([encoder.finish()]);
    const values = new Float32Array((await probeBuffer(world, scene.out, { size: 12 })).bytes);
    expect(await device.popErrorScope()).toBeNull();
    return { front: values[0]!, back: values[1]!, sunCovered: values[2]! };
}

test("a cullMode none row writes its back face's prepass depth and casts from it", async () => {
    const back = await depths("back");
    // Premise: a culled back face writes neither the view prepass nor the sun atlas.
    expect(back.front).toBeGreaterThan(0);
    expect(back.back).toBe(0);
    expect(back.sunCovered).toBe(0);
    const none = await depths("none");
    console.log(
        `cull none: back prepass depth ${none.back} (front ${back.front}), sun texels ${none.sunCovered}`,
    );
    expect(none.back).toBeCloseTo(back.front, 6);
    expect(none.sunCovered).toBeGreaterThan(0);
});

const switching = gpuScene(import.meta.path, { defaults: false, plugins: cullPlugins }, (world) => {
    const mesh = quad(world, "cull-switch-quad");
    const subject = world.create();
    world.add(subject, Transform);
    world.add(subject, MeshInstance, { mesh });
    const row = world.resource(Plain).add({ color: d.vec4f(0, 0, 1, 1) });
    world.add(subject, MeshMaterial, row);
    return { row, front: camera(world, 5), back: camera(world, -5) };
});

test("rows that only cull back faces add no draws, keys or indirect records, and a row switching cullMode redraws in that frame", async () => {
    const { world, scene } = switching();
    const plain = world.resource(Plain);
    world.step(0);
    const argsBytes = () => world.gpu.root.unwrap(world.resource(MeshDrawBuffers).drawArgs!).size;
    const culled = () => {
        let count = 0;
        world.resource(Draws).forEach((draw) => {
            if ((draw.cullMode ?? 0) !== 0) count++;
        });
        return count;
    };
    const backOnly = argsBytes();
    expect(culled()).toBe(0);
    expect((await pixels(world, scene.back)).center).toEqual(
        (await pixels(world, scene.back)).clear,
    );

    plain.update(scene.row, {}, { cullMode: "none" });
    world.step(0);
    expect((await pixels(world, scene.back)).center).toEqual(BLUE);
    expect(argsBytes()).toBe(backOnly * 2);
    expect(culled()).toBeGreaterThan(0);

    plain.update(scene.row, {}, { cullMode: "back" });
    world.step(0);
    const restored = await pixels(world, scene.back);
    expect(restored.center).toEqual(restored.clear);
    expect((await pixels(world, scene.front)).center).toEqual(BLUE);

    plain.update(scene.row, {}, { cullMode: "front" });
    world.step(0);
    expect((await pixels(world, scene.back)).center).toEqual(BLUE);
    const hidden = await pixels(world, scene.front);
    expect(hidden.center).toEqual(hidden.clear);
    expect(argsBytes()).toBe(backOnly * 3);

    for (const key of ["0-none", "0-front"])
        expectWarning(
            new RegExp(
                `Pipeline "standard-CullPlain-${key}-1x" owned by "CorePipeline/1" created after the precompile drain\\.`,
            ),
        );
});

const producer = gpuScene(import.meta.path, { defaults: false, plugins: cullPlugins }, (world) => {
    const mesh = quad(world, "cull-producer-quad");
    const row = world.resource(Plain).add({ color: d.vec4f(0, 0, 1, 1) }, { cullMode: "none" });
    const table = registerProducerTable(world, {
        name: "cull-producer",
        capacity: 1,
        meshes: [mesh],
        materials: [row],
    });
    table.placements.write([{ pos: d.vec3f(0), quat: d.vec4f(0, 0, 0, 1), scale: d.vec3f(1) }]);
    table.meshInstances.write([
        {
            mesh,
            materialType: row.type,
            material: row.material,
            alphaMode: row.alphaMode,
            alphaCutoff: row.alphaCutoff,
            flags: 0,
            cullMode: row.cullMode,
            cullBounds: d.vec4f(0, 0, 0, -1),
        },
    ]);
    table.count.write(1);
    return { back: camera(world, -5) };
});

test("a producer row with its handle's cullMode none draws its back face", async () => {
    const { world, scene } = producer();
    world.step(0);
    expect((await pixels(world, scene.back)).center).toEqual(BLUE);
});

const malformed = gpuScene(import.meta.path, { defaults: false, plugins: cullPlugins }, (world) => {
    const mesh = quad(world, "cull-malformed-quad");
    const row = world.resource(Plain).add({ color: d.vec4f(0, 0, 1, 1) });
    const table = registerProducerTable(world, {
        name: "cull-malformed",
        capacity: 1,
        meshes: [mesh],
        materials: [row],
    });
    table.placements.write([{ pos: d.vec3f(0), quat: d.vec4f(0, 0, 0, 1), scale: d.vec3f(1) }]);
    table.count.write(1);
    return { mesh, row, table, front: camera(world, 5) };
});

test("a producer row whose cullMode is above 2 draws nothing, including codes whose multiple wraps to a declared pair", async () => {
    const { world, scene } = malformed();
    const seen = async (cullMode: number) => {
        scene.table.meshInstances.write([
            {
                mesh: scene.mesh,
                materialType: scene.row.type,
                material: scene.row.material,
                alphaMode: scene.row.alphaMode,
                alphaCutoff: scene.row.alphaCutoff,
                flags: 0,
                cullMode,
                cullBounds: d.vec4f(0, 0, 0, -1),
            },
        ]);
        world.step(0);
        return pixels(world, scene.front);
    };
    expect((await seen(0)).center).toEqual(BLUE);
    for (const code of [3, 0x80000000, 0xaaaaaaab]) {
        const drawn = await seen(code);
        expect(drawn.center, `cullMode ${code}`).toEqual(drawn.clear);
    }
});

// Nine varyings fill the default 16 inter-stage variables beside the seven engine ones.
const nineVaryings = Object.fromEntries(
    Array.from({ length: 9 }, (_, i) => [`v${i}`, d.f32]),
) as Record<`v${number}`, d.F32>;
const Nine = materialType({
    name: "CullNineVaryings",
    parameters: ColorParameters,
    defaults: { color: d.vec4f(1) },
    varyings: nineVaryings,
    vertex: (input) => {
        "use gpu";
        return {
            world: input.world,
            worldNormal: input.worldNormal,
            v0: 0,
            v1: 0,
            v2: 0,
            v3: 0,
            v4: 0,
            v5: 0,
            v6: 0,
            v7: 0,
            v8: 0,
        };
    },
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(ctx.v0 + ctx.v8, 0, 0, 1);
    },
});
const NinePlugin = MaterialPlugin(Nine);
const budgetApps: Awaited<ReturnType<typeof createTestGpuApp>>[] = [];
const budget = compileGpuFile(import.meta.path, async () => {
    const build = async (cullMode: GPUCullMode) => {
        try {
            const app = await createTestGpuApp({
                defaults: false,
                plugins: [
                    ...DEFAULT_PLUGINS,
                    NinePlugin,
                    {
                        name: `CullBudget(${cullMode})`,
                        dependencies: [...DEFAULT_PLUGINS, NinePlugin],
                        initialize(world) {
                            const subject = world.create();
                            world.add(subject, Transform);
                            world.add(subject, MeshInstance);
                            world.add(
                                subject,
                                MeshMaterial,
                                world.resource(Nine).add({ color: d.vec4f(1) }, { cullMode }),
                            );
                            camera(world, 5);
                        },
                    },
                ],
            });
            budgetApps.push(app);
            expect(app.world.gpu.device.limits.maxInterStageShaderVariables).toBe(16);
            return null;
        } catch (error) {
            return error;
        }
    };
    return { back: await build("back"), none: await build("none") };
});
afterAll(() => disposeGpuApps(budgetApps));

test("a cullMode none row counts frontFacing against the device's inter-stage limit", () => {
    expect(budget().back).toBeNull();
    expect(String(budget().none)).toContain(
        'standard: material "CullNineVaryings" uses 17 inter-stage variables (7 engine, 9 varyings, frontFacing); device.limits.maxInterStageShaderVariables is 16',
    );
});
