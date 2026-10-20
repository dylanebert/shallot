import { expect } from "bun:test";
import { MeshInstance } from "../core/mesh";
import {
    attachTexture,
    Camera,
    captureTexture,
    Tonemapping,
    TonemappingMethod,
} from "../core/rendering";
import { Transform } from "../core/transform";
import type { World } from "../engine";
import { Materials, MeshMaterial, StandardMaterial, StandardRenderer } from "../standard/rendering";

export const SIZE = 48;
/** The lower camera's clear, presented unmapped, as sRGB bytes. */
export const LOWER = [0x20, 0x40, 0x60, 255];

/**
 * A clear-only lower camera with a standard camera stacked over it on one texture, drawing a lone unlit
 * cube turned 45° about the view axis, so its edges cover pixels partially under MSAA.
 */
export function stackedScene(world: World, antialias: number, size = SIZE) {
    const lower = world.create();
    world.add(lower, Transform, { translation: [0, 0, 5, 0] });
    world.add(lower, Camera, { clearColor: 0x204060, antialias });
    world.add(lower, Tonemapping, { method: TonemappingMethod.None });
    const target = attachTexture(world, lower, { width: size, height: size });
    const upper = world.create();
    world.add(upper, Transform, { translation: [0, 0, 5, 0] });
    world.add(upper, Camera, { order: 1, antialias });
    world.add(upper, StandardRenderer);
    attachTexture(world, upper, target);
    const cube = world.create();
    world.add(cube, Transform, { rotation: [0, 0, Math.sin(Math.PI / 8), Math.cos(Math.PI / 8)] });
    world.add(cube, MeshInstance);
    world.add(
        cube,
        MeshMaterial,
        world
            .resource(Materials)
            .add(StandardMaterial({ baseColor: [0.8, 0.1, 0.05, 1], unlit: true })),
    );
    return { lower, upper, cube };
}

export async function stackedFrame(world: World, camera: number): Promise<Uint8ClampedArray> {
    world.gpu.device.pushErrorScope("validation");
    world.step(0);
    world.step(0);
    const error = await world.gpu.device.popErrorScope();
    expect(error?.message ?? null).toBeNull();
    return (await captureTexture(world, camera)).rgba;
}

export function pixel(rgba: Uint8ClampedArray, x: number, y: number): number[] {
    const i = (y * SIZE + x) * 4;
    return Array.from(rgba.subarray(i, i + 4));
}

/** Pixels in the frame's 6×6 corners, where the cube, its outline band and its edges never reach. */
export function corners(rgba: Uint8ClampedArray): number[][] {
    const out: number[][] = [];
    for (const [x0, y0] of [
        [0, 0],
        [SIZE - 6, 0],
        [0, SIZE - 6],
        [SIZE - 6, SIZE - 6],
    ])
        for (let y = y0!; y < y0! + 6; y++)
            for (let x = x0!; x < x0! + 6; x++) out.push(pixel(rgba, x, y));
    return out;
}
