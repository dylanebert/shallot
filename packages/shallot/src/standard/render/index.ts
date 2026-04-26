import {
    resource,
    onAdd,
    onRemove,
    capacity,
    write,
    type Plugin,
    type State,
    type System,
} from "../../engine";
import { uploadViewport } from "./viewport";
import { type BufferView, type GBuf, gbuf, view } from "../compute";
import { Compute, ComputePlugin } from "../compute";
import { Canvas, ViewHooks, ViewportPlugin, ActiveCamera } from "../viewport";
import { WorldTransform } from "../transforms";
import {
    Camera,
    Tonemap,
    FXAA,
    Vignette,
    Posterize,
    Dither,
    Shadows,
    Reflections,
    Haze,
    Sky,
    Moon,
    Stars,
    Clouds,
    Sun,
    Viewport,
    RenderTarget,
} from "./camera";
import { AmbientLight, DirectionalLight, PointLight } from "./light";
import {
    Part,
    Mesh,
    Dynamic,
    PartColors,
    PartSizes,
    PartShapes,
    PartSurfaces,
    PartPBR,
    PartEmission,
    PartVolumes,
    MeshGeometryData,
    createShapeAtlas,
    updateShapeAtlas,
    type ShapeAtlas,
    getMeshVersion,
    allocateDynamic,
    deallocateDynamic,
    clearMeshes,
} from "./mesh";
import { clearDefaultSurfaces } from "./surface";
import { createSceneBuffer, createSkyBuffer } from "./scene";
import { createPresentNode } from "./present";
import { createOverlayNode } from "./overlay";
import type { OverlayDraw } from "./pass";
import { createDataNode } from "./data";
import { createInstanceNode } from "./instance";
import {
    createBatching,
    createBatchComputeNode,
    uploadResolveInputs,
    singleU32,
    type Batching,
} from "./batch";
import { POINT_LIGHT_BUFFER_SIZE, POINT_LIGHT_STRIDE, packPointLights } from "./light";
import type { ComputeNode, ExecutionContext } from "../compute";

export {
    Camera,
    CameraMode,
    Tonemap,
    FXAA,
    Vignette,
    Posterize,
    Dither,
    Shadows,
    Reflections,
    Haze,
    Sky,
    Moon,
    Stars,
    Clouds,
    Sun,
    Viewport,
    RenderTarget,
} from "./camera";
export { ActiveCamera } from "../viewport";
export { AmbientLight, DirectionalLight, PointLight } from "./light";
export {
    Part,
    Mesh,
    Dynamic,
    MeshShape,
    Volume,
    mesh,
    meshRegistry,
    createBox,
    createSphere,
    createPlane,
    createCone,
    getMesh,
} from "./mesh";
export type { MeshData } from "./mesh";
export { SurfaceType, surface, surfaceRegistry, property } from "./surface";
export type { SurfaceData, Property, PropertyType } from "./surface";

export interface Render {
    width: number;
    height: number;
    entityCount: number;
    meshVersion: number;
    viewProj: Float32Array;
    scene: GPUBuffer;
    sky: GPUBuffer;
    matrices: GBuf;
    colors: BufferView;
    sizes: BufferView;
    pbr: BufferView;
    emission: BufferView;
    shapes: BufferView;
    surfaces: BufferView;
    data: GBuf;
    entityCountBuffer: BufferView;
    propertiesBuffer: GBuf;
    u32Buffer: GBuf;

    meshAtlas: ShapeAtlas;
    effects: {
        overlay: OverlayDraw[];
    };
    batching: Batching;
    instanceDataBuffer: GBuf | null;
    viewportCap?: (cameraEid: number, w: number, h: number) => { w: number; h: number };
    pointLightBuffer: GPUBuffer;
    pointLightData: [Float32Array, number];
    needsDepth: boolean;
}

export const Render = resource<Render>("render");

const RenderSystem: System = {
    group: "draw",
    annotations: { mode: "always" },
    first: true,

    update(state: State) {
        const render = Render.from(state);
        const compute = Compute.from(state);
        if (!render || !compute) return;

        const { device } = compute;

        const currentMeshVersion = getMeshVersion();
        if (currentMeshVersion !== render.meshVersion) {
            updateShapeAtlas(device, render.meshAtlas);
            render.meshVersion = currentMeshVersion;
        }

        render.entityCount = state.max + 1;
        const uploadCount = render.entityCount;

        let t0 = performance.now();
        device.queue.writeBuffer(
            render.matrices.buffer,
            0,
            WorldTransform.data as Float32Array<ArrayBuffer>,
            0,
            uploadCount * 16,
        );

        const PropBytes = capacity() * 16;
        write(device.queue, render.propertiesBuffer.buffer, 0, PartColors, uploadCount);
        write(device.queue, render.propertiesBuffer.buffer, PropBytes, PartSizes, uploadCount);
        write(device.queue, render.propertiesBuffer.buffer, PropBytes * 2, PartPBR, uploadCount);
        write(
            device.queue,
            render.propertiesBuffer.buffer,
            PropBytes * 3,
            PartEmission,
            uploadCount,
        );
        state.scheduler.reportCpu("Render/0:upload", performance.now() - t0);

        t0 = performance.now();
        uploadResolveInputs(
            device,
            render.batching,
            PartShapes,
            MeshGeometryData,
            PartSurfaces,
            PartVolumes,
            uploadCount,
        );

        singleU32[0] = uploadCount;
        device.queue.writeBuffer(
            render.u32Buffer.buffer,
            capacity() * 2 * 4,
            singleU32 as Uint32Array<ArrayBuffer>,
        );
        state.scheduler.reportCpu("Render/0:write", performance.now() - t0);
    },
};

export const RenderPlugin: Plugin = {
    name: "Render",
    systems: [RenderSystem],
    components: {
        Camera,
        Part,
        Mesh,
        Dynamic,
        AmbientLight,
        DirectionalLight,
        PointLight,
        Tonemap,
        FXAA,
        Vignette,
        Posterize,
        Dither,
        Shadows,
        Reflections,
        Haze,
        Sky,
        Moon,
        Stars,
        Clouds,
        Sun,
        Viewport,
    },
    relations: [RenderTarget],
    dependencies: [ComputePlugin, ViewportPlugin],

    async initialize(state: State, onProgress?: (progress: number) => void) {
        clearDefaultSurfaces();
        clearMeshes();

        const compute = Compute.from(state);
        if (!compute) return;

        const { device } = compute;

        const StorageDst = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
        const StorageDstSrc = StorageDst | GPUBufferUsage.COPY_SRC;

        const shapeAtlas = createShapeAtlas(device);
        const batching = createBatching(device);

        const matrices = gbuf(device, "matrices", StorageDstSrc, (c) => c * 64);
        const propertiesBuffer = gbuf(device, "properties", StorageDstSrc, (c) => c * 64);
        const u32Buffer = gbuf(device, "u32-props", StorageDstSrc, (c) => c * 8 + 256);
        const data = gbuf(device, "data", StorageDstSrc, (c) => c * 64);

        const pointLightBuffer = device.createBuffer({
            label: "point-lights",
            size: POINT_LIGHT_BUFFER_SIZE,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const renderState: Render = {
            viewProj: new Float32Array(16),
            scene: createSceneBuffer(device),
            sky: createSkyBuffer(device),
            matrices,
            propertiesBuffer,
            u32Buffer,

            colors: view(
                propertiesBuffer,
                () => 0,
                (c) => c * 16,
            ),
            sizes: view(
                propertiesBuffer,
                (c) => c * 16,
                (c) => c * 16,
            ),
            pbr: view(
                propertiesBuffer,
                (c) => c * 32,
                (c) => c * 16,
            ),
            emission: view(
                propertiesBuffer,
                (c) => c * 48,
                (c) => c * 16,
            ),
            shapes: view(
                u32Buffer,
                () => 0,
                (c) => c * 4,
            ),
            surfaces: view(
                u32Buffer,
                (c) => c * 4,
                (c) => c * 4,
            ),
            entityCountBuffer: view(
                u32Buffer,
                (c) => c * 8,
                () => 256,
            ),
            data,
            entityCount: 1,
            meshVersion: getMeshVersion(),
            batching,
            effects: {
                overlay: [],
            },
            meshAtlas: shapeAtlas,
            width: 0,
            height: 0,
            instanceDataBuffer: null,
            pointLightBuffer,
            pointLightData: [new Float32Array(0), 0],
            needsDepth: false,
        };

        state.setResource(Render, renderState);
        state.setResource(ActiveCamera, { eid: -1 });

        const dataNode = createDataNode(renderState);
        compute.graph.add(dataNode);

        const instanceDataNode = createInstanceNode(renderState);
        compute.graph.add(instanceDataNode);

        state.observe(onAdd(Part), (eid: number) => {
            const idx = eid >>> 5;
            const b = renderState.batching;
            if (idx >= b.partMask.length) {
                const grown = new Uint32Array(capacity() >>> 5);
                grown.set(b.partMask);
                b.partMask = grown;
            }
            b.partMask[idx] |= 1 << (eid & 31);
            b.cullEntityCount++;
        });
        state.observe(onRemove(Part), (eid: number) => {
            renderState.batching.partMask[eid >>> 5] &= ~(1 << (eid & 31));
            renderState.batching.cullEntityCount--;
        });

        state.observe(onAdd(Dynamic), (eid: number) => {
            allocateDynamic(eid);
        });
        state.observe(onRemove(Dynamic), (eid: number) => {
            deallocateDynamic(eid);
        });

        const batchComputeNode = createBatchComputeNode(renderState);
        compute.graph.add(batchComputeNode);

        const pointLightUploadNode: ComputeNode = {
            name: "point-light-upload",
            scope: "frame",
            inputs: ["data"],
            outputs: ["point-light-data"],
            execute(_ctx: ExecutionContext) {
                const camEid = ActiveCamera.from(state)?.eid ?? -1;
                const cameraShadows = camEid >= 0 && state.hasComponent(camEid, Shadows);
                renderState.pointLightData = packPointLights(state, cameraShadows);
                const [lights, count] = renderState.pointLightData;
                if (count > 0) {
                    _ctx.device.queue.writeBuffer(
                        renderState.pointLightBuffer,
                        0,
                        lights.buffer,
                        lights.byteOffset,
                        count * POINT_LIGHT_STRIDE * 4,
                    );
                }
            },
        };
        compute.graph.add(pointLightUploadNode);

        compute.graph.add(
            createOverlayNode({
                overlays: renderState.effects.overlay,
                hasDepthWriter: (() => {
                    const cache = new Map<string, boolean>();
                    return (subGraph: string) => {
                        let cached = cache.get(subGraph);
                        if (cached !== undefined) return cached;
                        cached = false;
                        const sg = compute.graph.subGraphs.get(subGraph);
                        if (sg) {
                            for (const node of sg.nodes.values()) {
                                for (const out of node.outputs) {
                                    if (out === "depth") {
                                        cached = true;
                                        break;
                                    }
                                }
                                if (cached) break;
                            }
                        }
                        cache.set(subGraph, cached);
                        return cached;
                    };
                })(),
            }),
        );

        compute.graph.add(createPresentNode(renderState.scene));

        onProgress?.(1);
    },

    async warm(state: State) {
        const activeCameras: number[] = [];
        for (const eid of state.query([Camera])) {
            if (Camera.active[eid]) activeCameras.push(eid);
        }
        if (activeCameras.length > 0) {
            const canvasEntities: number[] = [];
            for (const eid of state.query([Canvas])) {
                canvasEntities.push(eid);
            }
            if (canvasEntities.length === 0) {
                const eid = state.addEntity();
                state.addComponent(eid, Canvas);
                canvasEntities.push(eid);
            }
            for (const eid of activeCameras) {
                const target = state.getFirstRelationTarget(eid, RenderTarget);
                if (target >= 0) continue;
                if (canvasEntities.length === 1) {
                    state.addRelation(eid, RenderTarget, canvasEntities[0]);
                }
            }
        }

        const hooks = ViewHooks.from(state);
        if (hooks) {
            hooks.push((state, canvasEid, view) => {
                const activeCamera = ActiveCamera.from(state);
                const render = Render.from(state);
                const compute = Compute.from(state);
                if (!compute) return;

                let cameraEid = -1;
                for (const eid of state.query([Camera])) {
                    if (!Camera.active[eid]) continue;
                    const target = state.getFirstRelationTarget(eid, RenderTarget);
                    if (target === canvasEid) {
                        cameraEid = eid;
                        break;
                    }
                }
                if (cameraEid < 0) return;

                if (activeCamera) activeCamera.eid = cameraEid;

                if (render) {
                    uploadViewport(compute.device, render, state, cameraEid, view);
                }
            });
        }
    },
};
