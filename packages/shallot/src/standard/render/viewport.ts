import type { State } from "../../engine";
import type { View } from "../viewport";
import {
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
    uploadCamera,
    uploadSky,
    sceneStaging,
    sceneStagingU32,
} from "./camera";
import {
    AmbientLight,
    DirectionalLight,
    PointLight,
    packLightUniforms,
    MAX_RASTER_POINT_LIGHTS,
} from "./light";
import { ensureTextures } from "./scene";
import type { Render } from "./index";

const ambientData = { color: 0x000000, intensity: 0 };
const directionalData = {
    color: 0x000000,
    intensity: 0,
    directionX: 0,
    directionY: -1.0,
    directionZ: 0,
};

export function uploadViewport(
    device: GPUDevice,
    render: Render,
    state: State,
    cameraEid: number,
    view: View,
): void {
    const { width, height } = view.element ?? view;

    let renderWidth = width;
    let renderHeight = height;

    if (state.hasComponent(cameraEid, Viewport)) {
        const vw = Viewport.width[cameraEid];
        const vh = Viewport.height[cameraEid];
        if (vw > 0 && vh > 0) {
            renderWidth = vw;
            renderHeight = vh;
        } else if (vh > 0 && height > 0) {
            renderHeight = vh;
            renderWidth = Math.max(1, Math.round(vh * (width / height)));
        } else if (vw > 0 && width > 0) {
            renderWidth = vw;
            renderHeight = Math.max(1, Math.round(vw * (height / width)));
        }
    }

    if (render.viewportCap) {
        const cap = render.viewportCap(cameraEid, renderWidth, renderHeight);
        renderWidth = cap.w;
        renderHeight = cap.h;
    }

    render.width = renderWidth;
    render.height = renderHeight;

    const needsDepth = render.needsDepth;
    const needsMask = render.effects.overlay.length > 0 || needsDepth;

    ensureTextures(
        device,
        renderWidth,
        renderHeight,
        view.textures,
        view.textureViews,
        needsDepth,
        needsMask,
    );

    const dirLid = state.only([DirectionalLight]);
    const dirShadows = dirLid < 0 || DirectionalLight.shadows[dirLid] !== 0;
    const hasShadows = state.hasComponent(cameraEid, Shadows) && dirShadows;

    const shadowSoftness = hasShadows ? (Shadows.softness[cameraEid] ?? 0) : 0;
    const shadowSamples = hasShadows ? Math.max(1, Shadows.samples[cameraEid] ?? 1) : 0;

    const reflectionEnabled = state.hasComponent(cameraEid, Reflections) ? 1 : 0;

    const hazeEid = state.only([Haze]);
    const hazeParams =
        hazeEid >= 0 ? { density: Haze.density[hazeEid], color: Haze.color[hazeEid] } : undefined;

    const skyEid = state.only([Sky]);
    const skyParams =
        skyEid >= 0
            ? { zenith: Sky.zenith[skyEid], horizon: Sky.horizon[skyEid], band: Sky.band[skyEid] }
            : undefined;

    const moonEid = state.only([Moon]);
    const moonParams =
        moonEid >= 0
            ? {
                  phase: Moon.phase[moonEid],
                  opacity: Moon.opacity[moonEid],
                  azimuth: Moon.azimuth[moonEid],
                  elevation: Moon.elevation[moonEid],
              }
            : undefined;

    const starsEid = state.only([Stars]);
    const starsParams =
        starsEid >= 0
            ? { intensity: Stars.intensity[starsEid], amount: Stars.amount[starsEid] }
            : undefined;

    const cloudsEid = state.only([Clouds]);
    const cloudsParams =
        cloudsEid >= 0
            ? {
                  coverage: Clouds.coverage[cloudsEid],
                  density: Clouds.density[cloudsEid],
                  height: Clouds.height[cloudsEid],
                  color: Clouds.color[cloudsEid],
              }
            : undefined;

    const sunEid = state.only([Sun]);
    const sunParams =
        sunEid >= 0
            ? {
                  size: Sun.size[sunEid],
                  color: Sun.color[sunEid],
                  glow: Sun.glow[sunEid],
                  azimuth: Sun.azimuth[sunEid],
                  elevation: Sun.elevation[sunEid],
              }
            : undefined;

    uploadCamera(
        render.viewProj,
        cameraEid,
        renderWidth,
        renderHeight,
        shadowSoftness,
        shadowSamples,
        reflectionEnabled,
        render.entityCount,
    );

    ambientData.color = 0x000000;
    ambientData.intensity = 0;
    directionalData.color = 0x000000;
    directionalData.intensity = 0;
    directionalData.directionX = 0;
    directionalData.directionY = -1.0;
    directionalData.directionZ = 0;

    const ambientEid = state.only([AmbientLight]);
    if (ambientEid >= 0) {
        ambientData.color = AmbientLight.color[ambientEid];
        ambientData.intensity = AmbientLight.intensity[ambientEid];
    }

    if (dirLid >= 0) {
        directionalData.color = DirectionalLight.color[dirLid];
        directionalData.intensity = DirectionalLight.intensity[dirLid];
        directionalData.directionX = DirectionalLight.directionX[dirLid];
        directionalData.directionY = DirectionalLight.directionY[dirLid];
        directionalData.directionZ = DirectionalLight.directionZ[dirLid];
    }

    const lightUniforms = packLightUniforms(ambientData, directionalData);
    sceneStaging.set(lightUniforms, 48);

    sceneStaging[76] = state.time.elapsed;
    let pointLightCount = 0;
    for (const _eid of state.query([PointLight])) {
        pointLightCount++;
        if (pointLightCount >= MAX_RASTER_POINT_LIGHTS) break;
    }
    sceneStagingU32[77] = pointLightCount;
    sceneStaging[78] = hasShadows && dirLid >= 0 ? DirectionalLight.shadows[dirLid] : 0;

    const hasTonemap = state.hasComponent(cameraEid, Tonemap);
    sceneStaging[80] = hasTonemap ? Tonemap.exposure[cameraEid] : 1.0;
    if (state.hasComponent(cameraEid, Vignette)) {
        sceneStaging[81] = Vignette.strength[cameraEid];
        sceneStaging[82] = Vignette.inner[cameraEid];
        sceneStaging[83] = Vignette.outer[cameraEid];
    } else {
        sceneStaging[81] = 0;
        sceneStaging[82] = 0;
        sceneStaging[83] = 1;
    }
    sceneStaging[84] = state.hasComponent(cameraEid, Posterize) ? Posterize.bands[cameraEid] : 0;
    sceneStaging[85] = state.hasComponent(cameraEid, Dither) ? Dither.strength[cameraEid] : 0;
    sceneStagingU32[86] = hasTonemap ? 1 : 0;
    sceneStagingU32[87] = state.hasComponent(cameraEid, FXAA) ? 1 : 0;

    device.queue.writeBuffer(render.scene, 0, sceneStaging as Float32Array<ArrayBuffer>);

    uploadSky(
        device,
        render.sky,
        hazeParams,
        skyParams,
        moonParams,
        starsParams,
        cloudsParams,
        sunParams,
    );
}
