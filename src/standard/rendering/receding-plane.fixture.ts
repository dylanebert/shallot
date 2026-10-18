// A long textured ground plane under a camera looking along it: the receding minification case.
// Inputs are generated here so the regression and the diagnostic need no assets or image decoder.

import { Meshes, MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    captureTexture,
    ImageBindings,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import type { World } from "../../engine";
import {
    type MaterialHandle,
    Materials,
    MeshMaterial,
    StandardMaterial,
    StandardRenderer,
} from ".";

/**
 * Diagnostic texture, varying along U only: 16-texel bands of two tones, crossed every 4 texels by a
 * 1-texel line in the other band's tone. Every rendered row then crosses the same content, so its
 * contrast follows the mip content sampled there. The lines average away at level 2, the bands at 5.
 */
export const IMAGE_SIZE = 1024;
export const TILE = 16;
export const TONES = [55, 200] as const;
/** Plane half-extent in world units; UVs span 0..1 over it, 64 texels per unit. */
export const HALF = 8;
export const EYE_HEIGHT = 0.5;
/** Camera pitch below the horizon, degrees; the vertical field of view is Camera's default 60. */
export const PITCH = 8;

/** sRGB RGBA bytes of the diagnostic texture's base level. */
export function diagnosticPixels(size = IMAGE_SIZE): Uint8Array {
    const pixels = new Uint8Array(size * size * 4);
    for (let y = 0; y < size; y++)
        for (let x = 0; x < size; x++) {
            const light = (x / TILE) & 1;
            const value = TONES[x % 4 === 0 ? 1 - light : light];
            pixels.set([value, value, value, 255], (y * size + x) * 4);
        }
    return pixels;
}

/** Per-level solid colours, so a capture shows which mip level each pixel selected. */
export const LEVEL_COLOURS = [
    [255, 0, 0],
    [255, 160, 0],
    [230, 230, 0],
    [0, 200, 0],
    [0, 200, 220],
    [0, 60, 255],
    [160, 0, 255],
    [255, 0, 200],
    [255, 255, 255],
    [128, 128, 128],
] as const;

const srgbToLinear = (byte: number) => {
    const c = byte / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
};
const linearToSrgb = (c: number) =>
    Math.round(255 * (c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055));

/** The level below `pixels`: each texel the linear-light mean of its 2×2 parent, as Shallot's blit. */
export function boxLevel(pixels: Uint8Array, size: number): Uint8Array {
    const half = size >> 1;
    const out = new Uint8Array(half * half * 4);
    for (let y = 0; y < half; y++)
        for (let x = 0; x < half; x++)
            for (let c = 0; c < 4; c++) {
                let sum = 0;
                for (const [dx, dy] of [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [1, 1],
                ]) {
                    const byte = pixels[((2 * y + dy) * size + 2 * x + dx) * 4 + c];
                    sum += c === 3 ? byte / 255 : srgbToLinear(byte);
                }
                out[(y * half + x) * 4 + c] =
                    c === 3 ? Math.round((sum / 4) * 255) : linearToSrgb(sum / 4);
            }
    return out;
}

let cachedChain: Uint8Array[] | undefined;

/** The diagnostic texture's full box-filtered mip chain, base level first. */
export function diagnosticChain(): readonly Uint8Array[] {
    if (cachedChain) return cachedChain;
    const chain = [diagnosticPixels()];
    for (let size = IMAGE_SIZE; size > 1; size >>= 1) chain.push(boxLevel(chain.at(-1)!, size));
    cachedChain = chain;
    return chain;
}

/**
 * Replaces the world's image array with `layers` copies of the diagnostic texture and its box-filtered
 * mip chain; `generate` instead fills each layer's chain from its base level. Layers listed in
 * `levelColoured` instead hold LEVEL_COLOURS per level.
 */
export function uploadDiagnosticImages(
    world: World,
    layers: number,
    options: {
        levelColoured?: readonly number[];
        generate?: (texture: GPUTexture, layer: number) => void;
    } = {},
): GPUTexture {
    const device = world.gpu.device;
    const levels = Math.log2(IMAGE_SIZE) + 1;
    const levelPixels = diagnosticChain();
    // The image array's descriptor plus COPY_SRC, so a check can read the levels back.
    const texture = device.createTexture({
        label: "receding-plane-images",
        size: [IMAGE_SIZE, IMAGE_SIZE, layers],
        format: "rgba8unorm-srgb",
        mipLevelCount: levels,
        usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.COPY_SRC |
            GPUTextureUsage.RENDER_ATTACHMENT,
    });
    world.own(texture);
    for (let layer = 0; layer < layers; layer++) {
        for (let level = 0; level < (options.generate ? 1 : levels); level++) {
            const size = IMAGE_SIZE >> level;
            device.queue.writeTexture(
                { texture, mipLevel: level, origin: [0, 0, layer] },
                levelPixels[level],
                { bytesPerRow: size * 4, rowsPerImage: size },
                [size, size, 1],
            );
        }
        options.generate?.(texture, layer);
    }
    for (const layer of options.levelColoured ?? []) {
        for (let level = 0; level < levels; level++) {
            const size = IMAGE_SIZE >> level;
            const pixels = new Uint8Array(size * size * 4);
            const [r, g, b] = LEVEL_COLOURS[level % LEVEL_COLOURS.length];
            for (let i = 0; i < size * size; i++) pixels.set([r, g, b, 255], i * 4);
            device.queue.writeTexture(
                { texture, mipLevel: level, origin: [0, 0, layer] },
                pixels,
                { bytesPerRow: size * 4, rowsPerImage: size },
                [size, size, 1],
            );
        }
    }
    world.resource(ImageBindings).images = texture;
    return texture;
}

/** Packed (px py pz u)(nx ny nz v) ground plane at y = 0, nearest edge at z = NEAR, receding along -Z. */
export function planeVertices(): Float32Array {
    const near = NEAR;
    const far = near - 2 * HALF;
    // prettier-ignore
    return new Float32Array([
        -HALF,
        0,
        near,
        0,
        0,
        1,
        0,
        0,
        HALF,
        0,
        near,
        1,
        0,
        1,
        0,
        0,
        HALF,
        0,
        far,
        1,
        0,
        1,
        0,
        1,
        -HALF,
        0,
        far,
        0,
        0,
        1,
        0,
        1,
    ]);
}

/** The plane's edge behind the eye, so the nearest visible rows are on it. */
export const NEAR = 1;

export const PLANE_INDICES = new Uint32Array([0, 1, 2, 0, 2, 3]);

export interface RecedingPlane {
    material: MaterialHandle;
    cameras: { eid: number; width: number; height: number }[];
}

/**
 * Spawns the plane with one unlit material and one camera per size, all at the same eye.
 * `vertices` defaults to planeVertices(); pass an importer's decoded mesh to compare it.
 */
export function spawnRecedingPlane(
    world: World,
    image: number,
    sizes: readonly { width: number; height: number }[],
    vertices = planeVertices(),
    indices = PLANE_INDICES,
): RecedingPlane {
    const mesh = world.resource(Meshes).add({ name: "receding-plane", vertices, indices });
    const material = world
        .resource(Materials)
        .add(StandardMaterial({ baseColorTexture: image, unlit: true }));
    const plane = world.create();
    world.add(plane, Transform);
    world.add(plane, MeshInstance, { mesh });
    world.add(plane, MeshMaterial, material);
    const half = (-PITCH * Math.PI) / 360;
    const cameras = sizes.map(({ width, height }) => {
        const eid = world.create();
        world.add(eid, Transform, {
            translation: [0, EYE_HEIGHT, 0, 0],
            rotation: [Math.sin(half), 0, 0, Math.cos(half)],
        });
        world.add(eid, Camera, { far: 1000 });
        world.storage(Camera).antialias.set(eid, 0);
        world.add(eid, StandardRenderer);
        world.add(eid, Tonemapping, { method: TonemappingMethod.None });
        attachTexture(world, eid, { width, height });
        return { eid, width, height };
    });
    return { material, cameras };
}

/** Renders `image` on the plane and reads back every camera's final RGBA. */
export async function capturePlane(
    world: World,
    plane: RecedingPlane,
    image: number,
): Promise<{ rgba: Uint8ClampedArray; width: number; height: number }[]> {
    world.resource(Materials).update(plane.material, { baseColorTexture: image });
    world.step(0);
    world.step(0);
    const out = [];
    for (const camera of plane.cameras) out.push(await captureTexture(world, camera.eid));
    return out;
}

/** Plane UV under target pixel coordinates (px, py) from the camera geometry, or undefined off it. */
export function planeUv(
    width: number,
    height: number,
    px: number,
    py: number,
): [number, number] | undefined {
    const tan = Math.tan(Math.PI / 6);
    const pitch = (PITCH * Math.PI) / 180;
    const x = ((2 * px) / width - 1) * tan * (width / height);
    const y = (1 - (2 * py) / height) * tan;
    const dy = y * Math.cos(pitch) - Math.sin(pitch);
    const dz = -y * Math.sin(pitch) - Math.cos(pitch);
    if (dy >= 0) return undefined;
    const t = -EYE_HEIGHT / dy;
    const uv: [number, number] = [(x * t + HALF) / (2 * HALF), (NEAR - dz * t) / (2 * HALF)];
    return uv[0] < 0 || uv[0] > 1 || uv[1] < 0 || uv[1] > 1 ? undefined : uv;
}

/** Isotropic level of detail at a pixel's centre: log2 of its larger footprint axis in base texels. */
export function planeLod(
    width: number,
    height: number,
    px: number,
    py: number,
): number | undefined {
    const c = planeUv(width, height, px + 0.5, py + 0.5);
    const r = planeUv(width, height, px + 1.5, py + 0.5);
    const b = planeUv(width, height, px + 0.5, py + 1.5);
    if (!c || !r || !b) return undefined;
    const x = Math.hypot(r[0] - c[0], r[1] - c[1]) * IMAGE_SIZE;
    const y = Math.hypot(b[0] - c[0], b[1] - c[1]) * IMAGE_SIZE;
    return Math.log2(Math.max(x, y));
}

type Readback = { rgba: Uint8ClampedArray; width: number; height: number };

/** Rows, top first, whose central half lies on the plane. */
export function planeRows({ width, height }: Readback): number[] {
    const rows: number[] = [];
    for (let y = 0; y < height; y++)
        if (
            planeUv(width, height, width / 4, y + 0.5) &&
            planeUv(width, height, (3 * width) / 4, y + 0.5)
        )
            rows.push(y);
    return rows;
}

/** A row's contrast: the standard deviation of red over its central half. */
export function rowContrast({ rgba, width }: Readback, y: number): number {
    let sum = 0;
    let square = 0;
    let n = 0;
    for (let x = width >> 2; x < width - (width >> 2); x++) {
        const value = rgba[(y * width + x) * 4];
        sum += value;
        square += value * value;
        n++;
    }
    const mean = sum / n;
    return Math.sqrt(Math.max(0, square / n - mean * mean));
}

/** Mean row contrast over plane rows whose centre level of detail lies in [low, high]. */
export function meanContrast(capture: Readback, low: number, high: number): number {
    let sum = 0;
    let rows = 0;
    for (const y of planeRows(capture)) {
        const lod = planeLod(capture.width, capture.height, capture.width / 2, y);
        if (lod === undefined || lod < low || lod > high) continue;
        sum += rowContrast(capture, y);
        rows++;
    }
    return rows === 0 ? 0 : sum / rows;
}

/**
 * The detail boundary along the receding direction: the steepest contrast change per unit of level
 * of detail between adjacent plane rows whose centre level of detail lies in [low, high]. Filtering
 * between levels changes contrast at a bounded rate; selecting one level steps it within a row or two.
 */
export function detailBoundary(capture: Readback, low: number, high: number) {
    let slope = 0;
    let row = -1;
    let previous: { y: number; lod: number; contrast: number } | undefined;
    for (const y of planeRows(capture)) {
        const lod = planeLod(capture.width, capture.height, capture.width / 2, y);
        if (lod === undefined || lod < low || lod > high) continue;
        const contrast = rowContrast(capture, y);
        if (previous?.y === y - 1) {
            const change = Math.abs(contrast - previous.contrast) / Math.abs(lod - previous.lod);
            if (change > slope) [slope, row] = [change, y];
        }
        previous = { y, lod, contrast };
    }
    return { slope, row, lod: planeLod(capture.width, capture.height, capture.width / 2, row) };
}
