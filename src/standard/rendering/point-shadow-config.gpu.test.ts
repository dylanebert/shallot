import { afterAll, expect, setDefaultTimeout, test } from "bun:test";
import { compileGpuFile } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { MeshInstance } from "../../core/mesh";
import {
    AmbientLight,
    attachTexture,
    Camera,
    captureTexture,
    PointLight,
} from "../../core/rendering";
import { createApp, Transform } from "../../engine";
import { rawDevice } from "../../engine/runtime";
import { StandardRenderer } from "./forward";
import { StandardRenderingPlugin } from "./index";
import { Materials, MeshMaterial, StandardMaterial } from "./material";
import { MeshRenderPlugin } from "./mesh-render";
import { PointShadows } from "./shadows";

setDefaultTimeout(CEILING.gpu);

// PointShadows is set after import and before createApp, as its JSDoc instructs; each arm holds its
// setting while its app builds and renders
const settings = {
    default: {},
    "casters 4": { casters: 4 },
    "atlas 1024": { atlas: 1024 },
} satisfies Record<string, Partial<typeof PointShadows>>;
type Arm = keyof typeof settings;
const order = Object.keys(settings) as Arm[];

async function withSetting<T>(arm: Arm, run: () => T | Promise<T>): Promise<T> {
    const saved = { ...PointShadows };
    Object.assign(PointShadows, settings[arm]);
    try {
        return await run();
    } finally {
        Object.assign(PointShadows, saved);
    }
}

type App = Awaited<ReturnType<typeof createApp>>;
const subjects = compileGpuFile(import.meta.path, async () => {
    const owner = await createApp({ defaults: false, plugins: [] });
    const device = rawDevice(owner.world.gpu.device);
    const apps = new Map<Arm, App>();
    for (const arm of order) {
        const app = await withSetting(arm, () =>
            createApp({
                defaults: false,
                plugins: [StandardRenderingPlugin, MeshRenderPlugin],
                device,
            }),
        );
        apps.set(arm, app);
    }
    return { owner, apps };
});

// a cube between a shadowed point light and a wall: the wall holds the cube's shadow
async function frame(arm: Arm): Promise<{ rgba: Uint8ClampedArray; error: string | null }> {
    const world = subjects().apps.get(arm)!.world;
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 6, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    attachTexture(world, camera, { width: 32, height: 32 });
    const material = world.resource(Materials).add(StandardMaterial());
    const wall = world.create();
    world.add(wall, Transform, { translation: [0, 0, -1, 0], scale: [20, 20, 0.2, 0] });
    world.add(wall, MeshInstance);
    world.add(wall, MeshMaterial, { material });
    const cube = world.create();
    world.add(cube, Transform, { translation: [0, 0, 1, 0] });
    world.add(cube, MeshInstance);
    world.add(cube, MeshMaterial, { material });
    world.add(world.create(), AmbientLight, { intensity: 0 });
    const light = world.create();
    world.add(light, Transform, { translation: [0.6, 0.6, 3, 0] });
    world.add(light, PointLight, { intensity: 20, range: 20 });
    world.storage(PointLight).shadowMapsEnabled.set(light, 1);
    const device = world.gpu.device;
    device.pushErrorScope("validation");
    // the shadow first appears on the second frame
    await withSetting(arm, () => {
        world.step(0);
        world.step(0);
    });
    const error = (await device.popErrorScope())?.message ?? null;
    const { rgba } = await captureTexture(world, camera);
    return { rgba, error };
}

let reference: Uint8ClampedArray | undefined;
async function defaultFrame(): Promise<Uint8ClampedArray> {
    if (!reference) {
        const { rgba, error } = await frame("default");
        expect(error).toBeNull();
        reference = rgba;
    }
    return reference;
}

test("the default point-shadow settings draw a shadowed wall without a validation error", async () => {
    const rgba = await defaultFrame();
    const red = (x: number, y: number) => rgba[(y * 32 + x) * 4];
    expect(red(12, 22)).toBeLessThan(red(28, 22));
});

test("PointShadows.casters set after import draws the default frame without a validation error", async () => {
    const expected = await defaultFrame();
    const { rgba, error } = await frame("casters 4");
    expect(error).toBeNull();
    expect(rgba).toEqual(expected);
});

test("PointShadows.atlas set after import draws without a validation error", async () => {
    const { error } = await frame("atlas 1024");
    expect(error).toBeNull();
});

afterAll(() => {
    const { owner, apps } = subjects();
    for (const app of [...apps.values()].reverse()) app.dispose();
    owner.dispose();
});
