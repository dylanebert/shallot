import {
    Camera,
    DirectionalLight,
    GlobalAmbientLight,
    lookAtRotation,
    Meshes,
    MeshInstance,
    MeshMaterial,
    type Plugin,
    type Resource,
    resetInterpolation,
    StandardRenderer,
    type System,
    Transform,
    type World,
} from "@dylanebert/shallot";
import {
    Orbit,
    OrbitPlugin,
    Sky,
    SkyPlugin,
    skyBackgroundHandle,
} from "@dylanebert/shallot/extras";
import { type MeshHandle, MeshPlugin, MeshQuant } from "@dylanebert/shallot/mesh";
import { beginComputePass, endComputePass } from "@dylanebert/shallot/runtime";
import {
    CameraBackground,
    type MaterialHandle,
    MeshPreprocessSystem,
    MeshRenderPlugin,
    StandardRenderingPlugin,
} from "@dylanebert/shallot/standard/rendering";
import * as d from "typegpu/data";
import { TerrainMaterial, TerrainMaterialPlugin } from "./material";
import { mesher } from "./mesher";
import { placement } from "./placement";
import { type Geometry, pool } from "./pool";
import { scatter } from "./scatter";
import {
    CONFIG,
    chunks,
    translation as chunkTranslation,
    chunkX,
    inRing,
    radii,
    SPEED,
    slotCount,
    type TerrainConfig,
} from "./terrain-data";

type Slot = {
    x: number;
    z: number;
    eid: number;
    instance: { mesh: MeshHandle };
    geometry: Geometry;
    dirty: boolean;
};
export const Terrain: Resource<{
    viewer: number;
    previousViewer: number;
    placement: ReturnType<typeof placement> | null;
    camera: number;
    turnovers: number;
    slots: Slot[];
    counts: GPUBuffer | null;
    details: GPUBuffer | null;
    detailCounts: GPUBuffer | null;
    scatterPipeline: GPUComputePipeline | null;
    scatterGroups: GPUBindGroup[];
    pipeline: GPUComputePipeline | null;
    emitPass: GPUComputePassDescriptor;
}> = {
    create: () => ({
        viewer: 0,
        previousViewer: 0,
        placement: null,
        camera: 0,
        turnovers: 0,
        slots: [],
        counts: null,
        details: null,
        detailCounts: null,
        scatterPipeline: null,
        scatterGroups: [],
        pipeline: null,
        emitPass: { label: "voxel:emit" },
    }),
};
const ZERO = new Uint32Array(1);

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
        const allFaces = inRing(x, slot.z, s.viewer, config);
        const geometry = slot.geometry;
        if (x !== slot.x) {
            world.remove(slot.eid, MeshInstance);
            meshes.remove(slot.instance.mesh);
            slot.instance.mesh = meshes.add(geometry.mesh);
            world.add(slot.eid, MeshInstance, slot.instance);
            resetInterpolation(world, slot.eid);
            s.turnovers++;
            slot.x = x;
            slot.dirty = true;
        }
        if (allFaces !== geometry.allFaces) {
            geometry.allFaces = allFaces;
            slot.dirty = true;
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
    const size = config.chunkSize;
    const span = (config.viewDistance * 2 + 1) * size;
    const placeDetails: System = {
        name: "place-details",
        group: "draw",
        before: [MeshPreprocessSystem],
        update(world) {
            const s = world.resource(Terrain);
            s.placement!.draw(
                s.previousViewer + (s.viewer - s.previousViewer) * world.time.fixedAlpha,
            );
        },
    };
    return {
        name: "Terrain",
        gpu: {},
        dependencies: [
            MeshPlugin,
            MeshRenderPlugin,
            StandardRenderingPlugin,
            TerrainMaterialPlugin,
            SkyPlugin,
            OrbitPlugin,
        ],
        recovery: (world) => ({
            snapshot: () => {
                const { viewer, turnovers } = world.resource(Terrain);
                return { viewer, turnovers };
            },
            restore: (snapshot) => {
                const saved = snapshot as { viewer: number; turnovers: number };
                const terrain = world.resource(Terrain);
                terrain.viewer = saved.viewer;
                terrain.previousViewer = saved.viewer;
                place(world, config);
                const instances = world.storage(MeshInstance).mesh;
                for (const slot of terrain.slots) {
                    instances.set(slot.eid, slot.instance.mesh);
                    slot.dirty = true;
                }
                terrain.turnovers = saved.turnovers;
            },
        }),
        initialize(world) {
            const s = world.resource(Terrain);
            const { device, root } = world.gpu;
            s.counts = device.createBuffer({
                size: slotCount(config) * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            });
            world.own(s.counts);
            const q = device.createBuffer({
                size: 48,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            world.own(q);
            s.detailCounts = device.createBuffer({
                size: slotCount(config) * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            });
            s.details = device.createBuffer({
                size: slotCount(config) * size * size * 8,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            });
            world.own(s.detailCounts);
            world.own(s.details);
            s.placement = placement(world, config, s.details, s.detailCounts);
            const { emit, quantData, height } = mesher(config);
            s.scatterPipeline = root.unwrap(
                root
                    .createComputePipeline({ compute: scatter(config, height) })
                    .$name("voxel:scatter"),
            );
            device.queue.writeBuffer(q, 0, quantData);
            const quant = root.createBuffer(d.arrayOf(MeshQuant, 1), q).$usage("storage");
            s.pipeline = root.unwrap(
                root.createComputePipeline({ compute: emit }).$name("voxel:emit"),
            );
            const geometries = pool(world, config, s.counts, quant, s.pipeline, slotCount(config));
            s.scatterGroups = geometries.map((geometry) =>
                device.createBindGroup({
                    layout: s.scatterPipeline!.getBindGroupLayout(0),
                    entries: [s.details!, s.detailCounts!, geometry.uniform].map(
                        (buffer, binding) => ({
                            binding,
                            resource: { buffer },
                        }),
                    ),
                }),
            );
            const { inner, outer } = radii(config);
            const material = world
                .resource(TerrainMaterial)
                .add({ ...TerrainMaterial.defaults!, inner, outer });
            for (const { x, z } of chunks(config)) {
                const i = s.slots.length;
                const geometry = geometries[i];
                geometry.allFaces = inRing(x, z, s.viewer, config);
                geometry.mesh.indirectIndexCount!.offset = i * 4;
                const handle = world.resource(Meshes).add(geometry.mesh);
                const eid = chunk(world, handle, material);
                s.slots.push({
                    x,
                    z,
                    eid,
                    instance: { mesh: handle },
                    geometry,
                    dirty: true,
                });
            }
            place(world, config);
            const ambient = world.resource(GlobalAmbientLight);
            ambient.color = 0xd8e4ff;
            ambient.brightness = 380;
            const sun = world.create();
            world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, 1, -0.65, -0.7) });
            world.add(sun, DirectionalLight, {
                color: 0xffe8cc,
                illuminance: 8000,
                shadowMapsEnabled: 1,
                maximumDistance: span * 1.65,
                firstCascadeFarBound: span * 0.625,
                depthBias: 0.04,
                shadowNormalBias: 1,
            });
            s.camera = world.create();
            world.add(s.camera, Camera, {
                clearColor: 0x89b6e9,
                fov: 55,
                far: span * 3,
                antialias: 1,
            });
            world.add(s.camera, CameraBackground, { background: skyBackgroundHandle(world) });
            const sky = world.create();
            world.add(sky, Sky, {
                hazeDensity: 0,
                cloudCoverage: 0,
                starIntensity: 0,
            });
            world.add(s.camera, StandardRenderer);
            world.add(s.camera, Transform);
            world.add(s.camera, Orbit, {
                pan: [0, 0, 0],
                distance: span * 0.99,
                yaw: Math.atan2(0.9, 1.2),
                pitch: 0.75,
                maxDistance: span * 2,
                flySpeed: size,
            });
        },
        systems: [
            {
                name: "scroll",
                group: "fixed",
                update(world) {
                    const s = world.resource(Terrain);
                    s.previousViewer = s.viewer;
                    s.viewer += SPEED / 60;
                    place(world, config);
                },
            },
            {
                name: "emit-chunks",
                group: "draw",
                before: [placeDetails, MeshPreprocessSystem],
                update(world) {
                    const s = world.resource(Terrain);
                    let pass: GPUComputePassEncoder | undefined;
                    for (let i = 0; i < s.slots.length; i++) {
                        const slot = s.slots[i];
                        if (!slot.dirty) continue;
                        s.placement!.chunk(i, slot.x, slot.z);
                        const geometry = slot.geometry!;
                        world.gpu.device.queue.writeBuffer(s.counts!, i * 4, ZERO);
                        world.gpu.device.queue.writeBuffer(s.detailCounts!, i * 4, ZERO);
                        geometry.input[0] = slot.x;
                        geometry.input[1] = slot.z;
                        geometry.input[2] = i;
                        geometry.input[3] = Number(geometry.allFaces);
                        world.gpu.device.queue.writeBuffer(geometry.uniform, 0, geometry.input);
                        if (!pass) {
                            s.emitPass.timestampWrites = world.gpu.span?.("voxel:emit");
                            pass = beginComputePass(world.frameEncoder()!, s.emitPass);
                        }
                        pass.setPipeline(s.pipeline!);
                        pass.setBindGroup(0, geometry.group);
                        pass.dispatchWorkgroups(size / 4, size / 4);
                        pass.setPipeline(s.scatterPipeline!);
                        pass.setBindGroup(0, s.scatterGroups[i]);
                        pass.dispatchWorkgroups(Math.ceil(size / 4), Math.ceil(size / 4));
                        slot.dirty = false;
                    }
                    if (pass) endComputePass(pass);
                },
            },
            placeDetails,
        ],
    };
}
export const TerrainPlugin = terrainPlugin(CONFIG);
