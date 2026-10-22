import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuScene } from "../../../scripts/gpu.fixture";
import { expectWarning } from "../../../scripts/test-console";
import { MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    CameraMode,
    captureTexture,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { offscreenTexture } from "../../core/rendering/view";
import { Transform } from "../../core/transform";
import type { World } from "../../engine";
import { probeTexture } from "../../engine/runtime";
import { DEFAULT_PLUGINS } from "..";
import { AlphaMode, Materials, MeshMaterial, StandardMaterial, StandardRenderer } from ".";

setDefaultTimeout(TIMEOUT.node);
const SIZE = 32;
const config = { defaults: false, plugins: [...DEFAULT_PLUGINS] };
function center(rgba: Uint8ClampedArray): number[] {
    const i = ((SIZE / 2) * SIZE + SIZE / 2) * 4;
    return Array.from(rgba.subarray(i, i + 4));
}
async function frame(world: World, eid: number): Promise<Uint8ClampedArray> {
    world.step(0);
    world.step(0);
    return (await captureTexture(world, eid)).rgba;
}
function stacked(
    world: World,
    antialias: number,
    clearColor: number,
    alphaMode: AlphaMode,
    alpha: number,
) {
    const lower = world.create();
    world.add(lower, Transform, { translation: [0, 0, 5, 0] });
    world.add(lower, Camera, { clearColor, antialias });
    world.add(lower, Tonemapping, { method: TonemappingMethod.None });
    const target = attachTexture(world, lower, { width: SIZE, height: SIZE });
    const upper = world.create();
    world.add(upper, Transform, { translation: [0, 0, 5, 0] });
    world.add(upper, Camera, { order: 1, antialias });
    world.add(upper, StandardRenderer);
    world.add(upper, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, upper, target);
    const cube = world.create();
    world.add(cube, Transform);
    world.add(cube, MeshInstance);
    world.add(
        cube,
        MeshMaterial,
        world
            .resource(Materials)
            .add(StandardMaterial({ baseColor: [1, 0, 0, alpha], unlit: true }), { alphaMode }),
    );
    return { lower, upper, cube };
}
const opaqueCases = [
    [AlphaMode.Opaque, 0.2],
    [AlphaMode.Mask(0.5), 0.6],
] as const;
const opaqueScenes = opaqueCases.flatMap(([mode, alpha]) =>
    [0x0000ff, 0x00ff00].map((clear) =>
        gpuScene(import.meta.path, config, (world) => stacked(world, 0, clear, mode, alpha)),
    ),
);
test("a camera stacked over another draws opaque and masked surfaces opaque whatever their material alpha", async () => {
    for (const subject of opaqueScenes) {
        const {
            world,
            scene: { lower, upper, cube },
        } = subject();
        expect(center(await frame(world, lower))).toEqual([255, 0, 0, 255]);
        for (const eid of [cube, upper, lower]) world.destroy(eid);
        world.step(0);
    }
});
const coverageScene = gpuScene(import.meta.path, config, (world) =>
    stacked(world, 1, 0x0000ff, AlphaMode.AlphaToCoverage, 0.5),
);
test("an AlphaToCoverage surface on a camera stacked over another covers it by coverage alone", async () => {
    const {
        world,
        scene: { lower, upper, cube },
    } = coverageScene();
    // Half the samples carry the red cube at full alpha, half the lower camera's blue.
    const [r, g, b, a] = center(await frame(world, lower));
    expect([g, a]).toEqual([0, 255]);
    expect(Math.abs(r! - 128)).toBeLessThanOrEqual(2);
    expect(Math.abs(r! + b! - 255)).toBeLessThanOrEqual(2);
    for (const eid of [cube, upper, lower]) world.destroy(eid);
    world.step(0);
});
const loneScene = gpuScene(import.meta.path, config, (world) => {
    const lone = world.create();
    world.add(lone, Transform, { translation: [0, 0, 5, 0] });
    world.add(lone, Camera, { mode: CameraMode.Orthographic, size: 2, clearColor: 0x0000ff });
    world.add(lone, StandardRenderer);
    world.add(lone, Tonemapping, { method: TonemappingMethod.None });
    const target = attachTexture(world, lone, { width: 64, height: 16 });
    // An alpha gradient: eight unit cubes across the view, alpha rising in eighths.
    const cubes: number[] = [];
    for (let i = 0; i < 8; i++) {
        const cube = world.create();
        world.add(cube, Transform, {
            translation: [(i - 3.5) * 1, 0, 0, 0],
            scale: [0.9, 0.9, 0.9, 0],
        });
        world.add(cube, MeshInstance);
        world.add(
            cube,
            MeshMaterial,
            world
                .resource(Materials)
                .add(StandardMaterial({ baseColor: [1, 0, 0, (i + 0.5) / 8], unlit: true }), {
                    alphaMode: AlphaMode.AlphaToCoverage,
                }),
        );
        cubes.push(cube);
    }
    return { lone, target, cubes };
});
test("a lone or bottom camera keeps hardware AlphaToCoverage, and only a camera stacked over another creates the coverage-mask variant", async () => {
    const {
        world,
        scene: { lone, target, cubes },
    } = loneScene();
    const device = world.gpu.device;
    const labels: string[] = [];
    const create = device.createRenderPipeline.bind(device);
    device.createRenderPipeline = (descriptor) => {
        labels.push(descriptor.label ?? "");
        return create(descriptor);
    };
    const stackedLabels = () => labels.filter((label) => label.endsWith("-stacked"));
    try {
        const alone = await frame(world, lone);
        // Hardware alphaToCoverage stores the material alpha in the samples it covers.
        const scene = await probeTexture(world, offscreenTexture(world, lone)!);
        const alphas = new Uint16Array(scene.bytes as ArrayBuffer);
        let fractional = 0;
        for (let i = 3; i < alphas.length; i += 4) if (alphas[i] !== 0x3c00) fractional++;
        expect(fractional).toBeGreaterThan(0);
        expect(stackedLabels()).toEqual([]);

        // Stacked over by a camera facing away, it is the bottom of its target and draws the same bytes.
        expectWarning(
            /Pipeline "standard-StandardMaterial-5-stacked" owned by "CorePipeline\/1" created after the precompile drain\./,
        );
        const away = world.create();
        world.add(away, Transform, { translation: [0, 0, 5, 0], rotation: [0, 1, 0, 0] });
        world.add(away, Camera, { order: 1, mode: CameraMode.Orthographic, size: 2 });
        world.add(away, StandardRenderer);
        attachTexture(world, away, target);
        expect(await frame(world, lone)).toEqual(alone);
        // The camera above draws the AlphaToCoverage materials through the coverage-mask variant.
        expect(stackedLabels()).toEqual(["standard-StandardMaterial-5-stacked"]);
        for (const eid of [...cubes, away, lone]) world.destroy(eid);
        world.step(0);
    } finally {
        device.createRenderPipeline = create;
    }
});
