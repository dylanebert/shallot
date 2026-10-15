import { afterAll, beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import {
    compileGpuFile,
    createTestGpuApp,
    disposeGpuApps,
    gpuRequirements,
} from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import {
    AmbientLight,
    attachTexture,
    Camera,
    captureTexture,
    DirectionalLight,
    PointLight,
    SpotLight,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { lookAtRotation, type World } from "../../engine";
import { StandardRenderer } from "./forward";
import { StandardRenderingPlugin } from "./index";
import { Materials, MeshMaterial, StandardMaterial } from "./material";
import { MeshRenderPlugin } from "./mesh-render";
import { DirectionalLightShadowMap, PointShadows } from "./shadows";

setDefaultTimeout(TIMEOUT.node);

const renderingExports = (await import("./index")) as unknown as Record<string, unknown>;
const shadowFilteringMethod = renderingExports["ShadowFilteringMethod"] as any;
const shadowFilteringMode = renderingExports["ShadowFilteringMethodMode"] as
    | Record<string, number>
    | undefined;
type LightKind = "directional" | "spot" | "point";
type Frame = { lit: Uint8ClampedArray; shadow: Uint8ClampedArray };
const IMAGE_SIDE = 192;
const apps: Awaited<ReturnType<typeof createTestGpuApp>>[] = [];
let device: GPUDevice;

function addScene(
    world: World,
    kind: LightKind,
    cameraX: number,
): { camera: number; light: number } {
    const camera = world.create();
    world.add(camera, Transform, { translation: [cameraX, 6, 8, 0] });
    world.storage(Transform).rotation.set(camera, ...lookAtRotation(cameraX, 6, 8, 0, 0, 0));
    world.add(camera, Camera);
    world.storage(Camera).antialias.set(camera, 0);
    world.add(camera, AmbientLight, { brightness: 0 });
    world.add(camera, StandardRenderer);
    attachTexture(world, camera, { width: IMAGE_SIDE, height: IMAGE_SIDE });

    const material = world
        .resource(Materials)
        .add(StandardMaterial({ baseColor: [0.8, 0.8, 0.8, 1] }));
    const floor = world.create();
    world.add(floor, Transform, { translation: [0, -0.1, 0, 0], scale: [12, 0.2, 12, 0] });
    world.add(floor, MeshInstance);
    world.add(floor, MeshMaterial, material);
    const box = world.create();
    world.add(box, Transform, { translation: [0, 0.65, 0, 0], scale: [1, 1.3, 1, 0] });
    world.add(box, MeshInstance);
    world.add(box, MeshMaterial, material);

    const light = world.create();
    if (kind === "directional") {
        world.add(light, Transform, { rotation: lookAtRotation(0, 0, 0, -0.65, -1, -0.35) });
        world.add(light, DirectionalLight, { illuminance: 15000 });
        world.storage(DirectionalLight).shadowMapsEnabled.set(light, 1);
    } else {
        world.add(light, Transform, { translation: [-3, 5, 2, 0] });
        if (kind === "spot") {
            world.storage(Transform).rotation.set(light, ...lookAtRotation(-3, 5, 2, 0, 0, 0));
            world.add(light, SpotLight, {
                intensity: 900000,
                range: 20,
                outerAngle: (55 * Math.PI) / 180,
            });
            world.storage(SpotLight).shadowMapsEnabled.set(light, 1);
        } else {
            world.add(light, PointLight, { intensity: 900000, range: 20 });
            world.storage(PointLight).shadowMapsEnabled.set(light, 1);
        }
    }
    return { camera, light };
}

function shadowFlag(world: World, kind: LightKind, light: number, enabled: number): void {
    if (kind === "directional")
        world.storage(DirectionalLight).shadowMapsEnabled.set(light, enabled);
    else if (kind === "spot") world.storage(SpotLight).shadowMapsEnabled.set(light, enabled);
    else world.storage(PointLight).shadowMapsEnabled.set(light, enabled);
}

async function image(kind: LightKind, method?: number, cameraX = 4): Promise<Frame> {
    const app = await createTestGpuApp({
        defaults: false,
        plugins: [StandardRenderingPlugin, MeshRenderPlugin],
        device,
        setup(world) {
            world.resource(DirectionalLightShadowMap).size = 1024;
            world.resource(PointShadows).atlas = 4096;
        },
    });
    const { camera, light } = addScene(app.world, kind, cameraX);
    if (shadowFilteringMethod && method !== undefined) {
        app.world.add(camera, shadowFilteringMethod, { method });
    }

    try {
        shadowFlag(app.world, kind, light, 0);
        app.world.step(0);
        app.world.step(0);
        const lit = (await captureTexture(app.world, camera)).rgba;

        shadowFlag(app.world, kind, light, 1);
        app.world.step(0);
        app.world.step(0);
        const shadow = (await captureTexture(app.world, camera)).rgba;
        return { lit, shadow };
    } finally {
        app.dispose();
    }
}

function luma(bytes: Uint8ClampedArray, pixel: number): number {
    const at = pixel * 4;
    return 0.2126 * bytes[at] + 0.7152 * bytes[at + 1] + 0.0722 * bytes[at + 2];
}

// The control frame has identical geometry and lighting but no shadow caster. Count only floor pixels
// between fully lit and fully shadowed values, so geometry silhouettes and the spot cone do not score.
function penumbraScore(frame: Frame): number {
    let score = 0;
    for (let pixel = 0; pixel < IMAGE_SIDE * IMAGE_SIDE; pixel++) {
        const lit = luma(frame.lit, pixel);
        if (lit <= 12) continue;
        const ratio = Math.max(0, Math.min(1, luma(frame.shadow, pixel) / lit));
        score += ratio * (1 - ratio);
    }
    return score;
}

function penumbraPixels(frame: Frame): number {
    let count = 0;
    for (let pixel = 0; pixel < IMAGE_SIDE * IMAGE_SIDE; pixel++) {
        const lit = luma(frame.lit, pixel);
        const shadow = luma(frame.shadow, pixel);
        if (lit > 12 && shadow > lit * 0.15 && shadow < lit * 0.85) count++;
    }
    return count;
}

function shadowedPixelChanges(a: Frame, b: Frame): number {
    let changed = 0;
    for (let pixel = 0; pixel < IMAGE_SIDE * IMAGE_SIDE; pixel++) {
        const aLit = luma(a.lit, pixel);
        const bLit = luma(b.lit, pixel);
        if (aLit <= 12 || bLit <= 12) continue;
        const aShadowed = luma(a.shadow, pixel) < aLit * 0.5;
        const bShadowed = luma(b.shadow, pixel) < bLit * 0.5;
        if (aShadowed !== bShadowed) changed++;
    }
    return changed;
}

const BASE_3X3_PENUMBRA_PIXELS = { directional: 9, spot: 6, point: 11 };
const BASE_3X3_PENUMBRA_SCORE = {
    directional: 2.9673019184128946,
    spot: 1.4602048375903547,
    point: 2.6851519884457042,
};
const BASE_3X3_CASCADE_MOVEMENT_PIXELS = 3;

const fixture = compileGpuFile(import.meta.path, async () => {
    const app = await createTestGpuApp({
        defaults: false,
        plugins: [
            {
                name: "ShadowFilteringNodeDevice",
                gpu: gpuRequirements([StandardRenderingPlugin, MeshRenderPlugin]),
            },
        ],
    });
    apps.push(app);
    device = app.world.gpu.device;
    return app;
});

beforeAll(() => {
    fixture();
});

afterAll(() => {
    fixture();
    return disposeGpuApps(apps);
});

for (const kind of ["directional", "spot", "point"] as const) {
    test(`${kind} box shadow has a hard Hardware2x2 edge and a Gaussian penumbra wider than the base 3x3`, async () => {
        if (!shadowFilteringMethod || !shadowFilteringMode) {
            const baselineFrame = await image(kind);
            const baseline = penumbraPixels(baselineFrame);
            const score = penumbraScore(baselineFrame);
            console.log(`base 3x3 ${kind} penumbra pixels: ${baseline}; score ${score}`);
            expect(baseline).toBe(BASE_3X3_PENUMBRA_PIXELS[kind]);
            expect(score).toBeCloseTo(BASE_3X3_PENUMBRA_SCORE[kind], 5);
            expect(shadowFilteringMethod).toBeDefined();
            return;
        }
        const hardware = await image(kind, shadowFilteringMode["Hardware2x2"]);
        const gaussian = await image(kind, shadowFilteringMode["Gaussian"]);
        const defaultMethod = await image(kind);
        const hardwarePixels = penumbraPixels(hardware);
        const gaussianPixels = penumbraPixels(gaussian);
        const hardwareScore = penumbraScore(hardware);
        const gaussianScore = penumbraScore(gaussian);
        console.log(
            `${kind} observations: base ${BASE_3X3_PENUMBRA_PIXELS[kind]}/${BASE_3X3_PENUMBRA_SCORE[kind]}, Hardware2x2 ${hardwarePixels}/${hardwareScore}, Gaussian ${gaussianPixels}/${gaussianScore}`,
        );
        expect(hardwareScore).toBeLessThan(BASE_3X3_PENUMBRA_SCORE[kind]);
        expect(gaussianScore).toBeGreaterThan(BASE_3X3_PENUMBRA_SCORE[kind]);
        expect(gaussianScore).toBeGreaterThan(hardwareScore);
        expect(defaultMethod.shadow).toEqual(gaussian.shadow);
        console.log(
            `${kind} penumbra pixels: base 3x3 ${BASE_3X3_PENUMBRA_PIXELS[kind]}, Hardware2x2 ${hardwarePixels}, Gaussian ${gaussianPixels}`,
        );
    });
}

test("Gaussian cascade camera sub-texel moves change no more shadowed pixels than the base 3x3", async () => {
    if (!shadowFilteringMethod || !shadowFilteringMode) {
        const first = await image("directional");
        let maxChanges = 0;
        for (let step = 1; step < 12; step++) {
            const next = await image("directional", undefined, 4 + step * 0.0137);
            maxChanges = Math.max(maxChanges, shadowedPixelChanges(first, next));
        }
        expect(maxChanges).toBe(BASE_3X3_CASCADE_MOVEMENT_PIXELS);
        console.log(`base 3x3 cascade sub-texel movement pixels: ${maxChanges}`);
        expect(shadowFilteringMethod).toBeDefined();
        return;
    }
    const first = await image("directional", shadowFilteringMode["Gaussian"]);
    let maxChanges = 0;
    for (let step = 1; step < 12; step++) {
        const next = await image("directional", shadowFilteringMode["Gaussian"], 4 + step * 0.0137);
        maxChanges = Math.max(maxChanges, shadowedPixelChanges(first, next));
    }
    expect(maxChanges).toBeLessThanOrEqual(BASE_3X3_CASCADE_MOVEMENT_PIXELS);
    console.log(
        `cascade sub-texel movement shadowed pixels: base 3x3 ${BASE_3X3_CASCADE_MOVEMENT_PIXELS}, Gaussian ${maxChanges}`,
    );
});
