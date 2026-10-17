import {
    Camera,
    DirectionalLight,
    GlobalAmbientLight,
    lookAtRotation,
    Materials,
    Meshes,
    MeshInstance,
    MeshMaterial,
    type Plugin,
    type Resource,
    resetInterpolation,
    StandardMaterial,
    StandardRenderer,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { type Mesh, type MeshHandle, MeshPlugin, MeshQuant } from "@dylanebert/shallot/mesh";
import {
    type MaterialHandle,
    MeshPreprocessSystem,
    MeshRenderPlugin,
    StandardRenderingPlugin,
} from "@dylanebert/shallot/standard/rendering";
import * as d from "typegpu/data";
import { mesher } from "./mesher";
import {
    CONFIG,
    capacity,
    chunks,
    translation as chunkTranslation,
    chunkX,
    SPEED,
    slotCount,
    type TerrainConfig,
} from "./terrain-data";

type Slot = {
    x: number;
    z: number;
    eid: number;
    instance: { mesh: MeshHandle };
    mesh: Mesh;
    uniform: GPUBuffer;
    input: Int32Array;
    group: GPUBindGroup;
    dirty: boolean;
};
export const Terrain: Resource<{
    viewer: number;
    camera: number;
    turnovers: number;
    slots: Slot[];
    counts: GPUBuffer | null;
    pipeline: GPUComputePipeline | null;
}> = {
    create: () => ({ viewer: 0, camera: 0, turnovers: 0, slots: [], counts: null, pipeline: null }),
};
const ZERO = new Uint32Array(1);
const EMIT_PASS = { label: "voxel:emit" };

function chunk(world: World, mesh: MeshHandle, material: MaterialHandle): number {
    const eid = world.create();
    world.add(eid, Transform);
    world.add(eid, MeshInstance, { mesh });
    world.add(eid, MeshMaterial, material);
    return eid;
}

function place(world: World, config: TerrainConfig): void {
    const s = world.resource(Terrain);
    const meshes = world.resource(Meshes);
    const translation = world.storage(Transform).translation;
    for (const slot of s.slots) {
        const x = chunkX(slot.x, s.viewer, config);
        if (x !== slot.x) {
            world.remove(slot.eid, MeshInstance);
            meshes.remove(slot.instance.mesh);
            slot.instance.mesh = meshes.add(slot.mesh);
            world.add(slot.eid, MeshInstance, slot.instance);
            slot.x = x;
            slot.dirty = true;
            resetInterpolation(world, slot.eid);
            s.turnovers++;
        }
        translation.set(
            slot.eid,
            chunkTranslation(slot.x, s.viewer, config),
            -config.height / 2,
            chunkTranslation(slot.z, 0, config),
        );
    }
}

export function terrainPlugin(config: TerrainConfig): Plugin {
    const Chunk = config.chunkSize;
    const Capacity = capacity(config);
    const Slots = slotCount(config);
    return {
        name: "Terrain",
        gpu: {},
        dependencies: [MeshPlugin, MeshRenderPlugin, StandardRenderingPlugin],
        recovery: (world) => ({
            snapshot: () => {
                const { viewer, turnovers } = world.resource(Terrain);
                return { viewer, turnovers };
            },
            restore: (snapshot) => {
                const saved = snapshot as { viewer: number; turnovers: number };
                const terrain = world.resource(Terrain);
                terrain.viewer = saved.viewer;
                place(world, config);
                terrain.turnovers = saved.turnovers;
            },
        }),
        initialize(world) {
            const s = world.resource(Terrain);
            const { device, root } = world.gpu;
            const buffer = (size: number, usage: GPUBufferUsageFlags) => {
                const value = device.createBuffer({ size, usage });
                world.own(value);
                return value;
            };
            s.counts = buffer(
                Slots * 4,
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            );
            const q = buffer(48, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
            const { emit, quantData } = mesher(config);
            device.queue.writeBuffer(q, 0, quantData);
            const quant = root.createBuffer(d.arrayOf(MeshQuant, 1), q).$usage("storage");
            s.pipeline = root.unwrap(
                root.createComputePipeline({ compute: emit }).$name("voxel:emit"),
            );
            const material = world.resource(Materials).add(
                StandardMaterial({
                    baseColor: [0.72, 0.4, 0.05, 1],
                    perceptualRoughness: 0.88,
                }),
            );
            for (const { x, z } of chunks(config)) {
                const i = s.slots.length;
                const vertices = buffer(Capacity * 4 * 16, GPUBufferUsage.STORAGE);
                const position = buffer(Capacity * 4 * 8, GPUBufferUsage.STORAGE);
                const indices = buffer(
                    Capacity * 6 * 4,
                    GPUBufferUsage.STORAGE | GPUBufferUsage.INDEX,
                );
                const uniform = buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                const mesh: Mesh = {
                    name: "chunk",
                    vertices: root
                        .createBuffer(d.arrayOf(d.vec4u, Capacity * 4), vertices)
                        .$usage("storage"),
                    position: root
                        .createBuffer(d.arrayOf(d.vec2u, Capacity * 4), position)
                        .$usage("storage"),
                    indices: root
                        .createBuffer(d.arrayOf(d.u32, Capacity * 6), indices)
                        .$usage("storage", "index"),
                    quant,
                    indexBase: 0,
                    indexCount: Capacity * 6,
                    indirectIndexCount: { buffer: s.counts, offset: i * 4 },
                    bounds: [
                        Chunk / 2,
                        config.height / 2,
                        Chunk / 2,
                        Math.hypot(Chunk, config.height, Chunk) / 2,
                    ],
                };
                const handle = world.resource(Meshes).add(mesh);
                const eid = chunk(world, handle, material);
                const group = device.createBindGroup({
                    layout: s.pipeline.getBindGroupLayout(0),
                    entries: [vertices, position, indices, s.counts, uniform].map(
                        (buffer, binding) => ({ binding, resource: { buffer } }),
                    ),
                });
                s.slots.push({
                    x,
                    z,
                    eid,
                    instance: { mesh: handle },
                    mesh,
                    uniform,
                    input: new Int32Array([x, z, i, 0]),
                    group,
                    dirty: true,
                });
            }
            place(world, config);
            const ambient = world.resource(GlobalAmbientLight);
            ambient.color = 0xd8e4ff;
            ambient.brightness = 380;
            const sun = world.create();
            world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, 1, -2, -0.7) });
            world.add(sun, DirectionalLight, {
                color: 0xffe8cc,
                illuminance: 8000,
                shadowMapsEnabled: 1,
                maximumDistance: 180,
            });
            s.camera = world.create();
            world.add(s.camera, Camera, { clearColor: 0x191816, fov: 55, far: 300 });
            world.add(s.camera, StandardRenderer);
            world.add(s.camera, Transform, {
                translation: [82, 76, 100],
                rotation: lookAtRotation(82, 76, 100, 0, -3, 0),
            });
        },
        systems: [
            {
                name: "scroll",
                group: "fixed",
                update(world) {
                    world.resource(Terrain).viewer += SPEED / 60;
                    place(world, config);
                },
            },
            {
                name: "emit-chunks",
                group: "draw",
                before: [MeshPreprocessSystem],
                update(world) {
                    const s = world.resource(Terrain);
                    let pass: GPUComputePassEncoder | undefined;
                    for (let i = 0; i < s.slots.length; i++) {
                        const slot = s.slots[i];
                        if (!slot.dirty) continue;
                        world.gpu.device.queue.writeBuffer(s.counts!, i * 4, ZERO);
                        slot.input[0] = slot.x;
                        slot.input[1] = slot.z;
                        world.gpu.device.queue.writeBuffer(slot.uniform, 0, slot.input);
                        pass ??= world.frameEncoder()!.beginComputePass(EMIT_PASS);
                        pass.setPipeline(s.pipeline!);
                        pass.setBindGroup(0, slot.group);
                        pass.dispatchWorkgroups(Chunk / 4, Chunk / 4);
                        slot.dirty = false;
                    }
                    pass?.end();
                },
            },
        ],
    };
}
export const TerrainPlugin = terrainPlugin(CONFIG);
