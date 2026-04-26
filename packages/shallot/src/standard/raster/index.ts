import type { Plugin, State } from "../../engine";
import { normalizeDirection, unpackColor } from "../../engine";
import type { ComputeNode, ExecutionContext } from "../compute";
import { Compute } from "../compute";
import { WorldTransform } from "../transforms";
import { Camera, Shadows, ActiveCamera, Sky } from "../render";
import { POINT_LIGHT_STRIDE } from "../render/core";
import { surfaceRegistry } from "../render";
import { Render, RenderPlugin, DirectionalLight } from "../render";
import { createCullPipeline } from "./cull";
import {
    createShadowAtlas,
    createShadowBuffer,
    createShadowUploadNode,
    createShadowForwardNode,
    createPointShadowUploadNode,
    createPointShadowForwardNode,
    ensurePointShadows,
    POINT_SHADOW_BUFFER_SIZE,
    type PointShadowState,
} from "./shadow";
import { resource } from "../../engine";
import { createRasterForwardNode } from "./forward";
import { clusterBufferSizes, createClusterCullNode } from "./cluster";

const DEFAULT_SHADOW_DISTANCE = 100;

export interface Raster {
    shadowAtlas: GPUTexture;
    shadowBuffer: GPUBuffer;
    pointShadowAtlas: GPUTexture;
    pointShadowBuffer: GPUBuffer;
    clusterParamsBuffer: GPUBuffer;
    clusterGridBuffer: GPUBuffer;
    lightIndexBuffer: GPUBuffer;
    bindGroupsDirty: boolean;
}

export const Raster = resource<Raster>("raster");

export const RasterPlugin: Plugin = {
    name: "Raster",
    systems: [],
    components: {},
    dependencies: [RenderPlugin],

    async initialize(state: State) {
        const compute = Compute.from(state);
        const render = Render.from(state);
        if (!compute || !render) return;

        const { device } = compute;

        let shadowAtlas: GPUTexture = device.createTexture({
            label: "shadow-atlas-placeholder",
            size: [1, 1],
            format: "depth32float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        let shadowAtlasAllocated = false;
        const shadowBuffer = createShadowBuffer(device);
        const pointShadowPlaceholderAtlas = device.createTexture({
            label: "point-shadow-atlas",
            size: [1, 1],
            format: "depth32float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        const pointShadowPlaceholderBuffer = device.createBuffer({
            label: "point-shadow",
            size: POINT_SHADOW_BUFFER_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const pointShadows: { current: PointShadowState | null } = { current: null };

        const MaxClusterWidth = 3840;
        const MaxClusterHeight = 2160;
        const clusterParamsBuffer = device.createBuffer({
            label: "cluster-params",
            size: 128,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        let clusterGridBuffer = device.createBuffer({
            label: "cluster-grid",
            size: 8,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        let lightIndexBuffer = device.createBuffer({
            label: "light-indices",
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        let clusterBuffersAllocated = false;

        const rasterState: Raster = {
            shadowAtlas,
            shadowBuffer,
            pointShadowAtlas: pointShadowPlaceholderAtlas,
            pointShadowBuffer: pointShadowPlaceholderBuffer,
            clusterParamsBuffer,
            clusterGridBuffer,
            lightIndexBuffer,
            bindGroupsDirty: false,
        };

        state.setResource(Raster, rasterState);

        const getActiveCamera = () => ActiveCamera.from(state)?.eid ?? -1;

        const getActiveClearColor = () => {
            const eid = getActiveCamera();
            if (eid >= 0) return unpackColor(Camera.clearColor[eid]);
            return { r: 0, g: 0, b: 0 };
        };

        const getSky = () => state.only([Sky]) >= 0;

        const getDirShadows = () => {
            const eid = getActiveCamera();
            if (eid < 0 || !state.hasComponent(eid, Shadows)) return false;
            const lid = state.only([DirectionalLight]);
            return lid >= 0 && DirectionalLight.shadows[lid] !== 0;
        };

        const getShadowDistance = () => {
            const eid = getActiveCamera();
            if (eid >= 0 && state.hasComponent(eid, Shadows)) {
                return Shadows.distance[eid];
            }
            return DEFAULT_SHADOW_DISTANCE;
        };

        const getCameraData = () => {
            const eid = getActiveCamera();
            if (eid >= 0) {
                return {
                    world: WorldTransform.data.subarray(eid * 16, eid * 16 + 16),
                    fov: Camera.fov[eid],
                    near: Camera.near[eid],
                    far: Camera.far[eid],
                    width: render.width,
                    height: render.height,
                };
            }
            return null;
        };

        const getLightDir = (): [number, number, number] => {
            const eid = state.only([DirectionalLight]);
            if (eid >= 0) {
                const [dx, dy, dz] = normalizeDirection(
                    DirectionalLight.directionX[eid],
                    DirectionalLight.directionY[eid],
                    DirectionalLight.directionZ[eid],
                );
                return [dx, dy, dz];
            }
            return [-0.5, -1.0, -0.5];
        };

        const sg = compute.graph.subGraph("raster");

        const shadowUploadNode = createShadowUploadNode(
            shadowBuffer,
            getCameraData,
            getLightDir,
            getDirShadows,
            getShadowDistance,
        );
        sg.add(shadowUploadNode);

        const ensureShadowAtlas = () => {
            if (!shadowAtlasAllocated) {
                shadowAtlasAllocated = true;
                shadowAtlas.destroy();
                shadowAtlas = createShadowAtlas(device);
                rasterState.shadowAtlas = shadowAtlas;
                rasterState.bindGroupsDirty = true;
            }
            return shadowAtlas;
        };

        const shadowNodes = createShadowForwardNode(
            render,
            shadowBuffer,
            ensureShadowAtlas,
            getDirShadows,
            surfaceRegistry.all,
        );
        for (const node of shadowNodes) sg.add(node);

        const getPointLights = (): [Float32Array, number] => render.pointLightData;

        const pointLightSetupNode: ComputeNode = {
            name: "point-light-setup",
            scope: "frame",
            inputs: ["point-light-data"],
            outputs: ["point-light-raster"],
            execute(ctx: ExecutionContext) {
                const [lights, count] = render.pointLightData;
                cachedLightCount = count;

                let hasShadowCasters = false;
                for (let i = 0; i < count; i++) {
                    if (lights[i * POINT_LIGHT_STRIDE + 7] >= 0) {
                        hasShadowCasters = true;
                        break;
                    }
                }
                if (hasShadowCasters && !pointShadows.current) {
                    const ps = ensurePointShadows(ctx.device, pointShadows);
                    pointShadowPlaceholderAtlas.destroy();
                    pointShadowPlaceholderBuffer.destroy();
                    rasterState.pointShadowAtlas = ps.atlas;
                    rasterState.pointShadowBuffer = ps.buffer;
                    rasterState.bindGroupsDirty = true;
                }

                if (count > 0 && !clusterBuffersAllocated) {
                    clusterBuffersAllocated = true;
                    clusterGridBuffer.destroy();
                    lightIndexBuffer.destroy();
                    const sizes = clusterBufferSizes(MaxClusterWidth, MaxClusterHeight);
                    clusterGridBuffer = ctx.device.createBuffer({
                        label: "cluster-grid",
                        size: sizes.gridSize,
                        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                    });
                    lightIndexBuffer = ctx.device.createBuffer({
                        label: "light-indices",
                        size: sizes.indexSize,
                        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                    });
                    rasterState.clusterGridBuffer = clusterGridBuffer;
                    rasterState.lightIndexBuffer = lightIndexBuffer;
                    rasterState.bindGroupsDirty = true;
                }
            },
        };
        sg.add(pointLightSetupNode);

        let cachedLightCount = 0;
        const clusterCullNode = createClusterCullNode(
            rasterState,
            render.pointLightBuffer,
            () => getActiveCamera(),
            () => cachedLightCount,
            MaxClusterWidth,
            MaxClusterHeight,
        );
        sg.add(clusterCullNode);

        const pointShadowUploadNode = createPointShadowUploadNode(
            getPointLights,
            () => pointShadows.current?.buffer ?? null,
        );
        sg.add(pointShadowUploadNode);

        const pointShadowRenderNode = createPointShadowForwardNode(
            render,
            () => pointShadows.current,
            getPointLights,
            surfaceRegistry.all,
        );
        sg.add(pointShadowRenderNode);

        const cull = createCullPipeline({
            matrices: render.matrices,
            sizes: render.sizes,
            batching: render.batching,
            viewProj: render.viewProj,
        });
        for (const node of cull.nodes) sg.add(node);

        const forwardNode = createRasterForwardNode(
            render,
            rasterState,
            surfaceRegistry.all,
            getActiveClearColor,
            getSky,
        );
        sg.add(forwardNode);
    },
};
