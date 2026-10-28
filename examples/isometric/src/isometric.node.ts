import { afterAll, beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { Camera, Devices, MeshInstance, Transform } from "@dylanebert/shallot";
import { Orbit } from "@dylanebert/shallot/extras";
import { attachTexture, captureTexture } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { Isometric, IsometricPlugin } from "./isometric";
import { layout, POPULATION, SEED } from "./layout";

setDefaultTimeout(TIMEOUT.node);
let app: Awaited<ReturnType<typeof openGpuApp>>;
beforeAll(async () => {
    app = await openGpuApp({
        plugins: [
            {
                name: "CaptureIsometric",
                dependencies: [IsometricPlugin],
                initialize(world) {
                    attachTexture(world, world.resource(Isometric).camera, {
                        width: 320,
                        height: 180,
                    });
                },
            },
        ],
    });
    console.info("isometric adapter", app.world.gpu.adapter);
});
afterAll(() => app?.dispose());
function step(count = 2) {
    for (let i = 0; i < count; i++) app.world.step(1 / 60);
}
function clear(r: number, g: number, b: number) {
    return Math.abs(r - 25) + Math.abs(g - 24) + Math.abs(b - 22) < 10;
}
async function frame() {
    step();
    return captureTexture(app.world, app.world.resource(Isometric).camera);
}

test("the default isometric frame draws buildings and an accent across its centre", async () => {
    expect([...app.world.query([MeshInstance])]).toHaveLength(POPULATION);
    const image = await frame();
    let covered = 0,
        accents = 0;
    for (let y = 60; y < 110; y++)
        for (let x = 100; x < 220; x++) {
            const i = (y * image.width + x) * 4;
            const [r, g, b] = image.rgba.slice(i, i + 3);
            covered += Number(Math.abs(r - 25) + Math.abs(g - 24) + Math.abs(b - 22) > 25);
            accents += Number(Math.max(r, g, b) - Math.min(r, g, b) > 35);
        }
    expect(covered / 6000).toBeGreaterThan(0.8);
    expect(accents / 6000).toBeGreaterThan(0.01);
});

test("scroll changes bounded size, not distance, and enlarges the buildings' footprint", async () => {
    const { world } = app,
        camera = world.resource(Isometric).camera;
    const orbit = world.storage(Orbit),
        cameras = world.storage(Camera),
        input = world.resource(Devices);
    const distance = () => {
        const t = world.storage(Transform).translation;
        return Math.hypot(
            t.x.get(camera) - orbit.pan.x.get(camera),
            t.y.get(camera) - orbit.pan.y.get(camera),
            t.z.get(camera) - orbit.pan.z.get(camera),
        );
    };
    const accents = (rgba: Uint8ClampedArray) => {
        let count = 0;
        for (let i = 0; i < rgba.length; i += 4)
            if (
                Math.max(rgba[i], rgba[i + 1], rgba[i + 2]) -
                    Math.min(rgba[i], rgba[i + 1], rgba[i + 2]) >
                35
            )
                count++;
        return count;
    };
    const before = await frame(),
        start = cameras.size.get(camera),
        d = distance();
    input.pointer.scroll = -100;
    step();
    input.pointer.scroll = 0;
    step(40);
    expect(cameras.size.get(camera)).toBeLessThan(start);
    expect(distance()).toBeCloseTo(d, 3);
    expect(accents((await frame()).rgba)).toBeGreaterThan(accents(before.rgba));
    for (const [scroll, bound] of [
        [-100000, orbit.minSize.get(camera)],
        [100000, orbit.maxSize.get(camera)],
    ]) {
        input.pointer.scroll = scroll;
        step();
        input.pointer.scroll = 0;
        step(60);
        expect(cameras.size.get(camera)).toBeCloseTo(bound, 3);
        expect(distance()).toBeCloseTo(d, 3);
    }
});

test("held arrow keys stop pitch at 25 and 60 degrees", () => {
    const { world } = app,
        camera = world.resource(Isometric).camera;
    const keys = world.resource(Devices).keys.held;
    for (const [key, degrees] of [
        ["ArrowUp", 60],
        ["ArrowDown", 25],
    ] as const) {
        keys.add(key);
        step(90);
        keys.delete(key);
        expect(world.storage(Orbit).pitch.get(camera)).toBeCloseTo((degrees * Math.PI) / 180, 5);
    }
});

test("inside a tall building the negative near retains pixels positive near clears", async () => {
    const { world } = app,
        camera = world.resource(Isometric).camera;
    const orbit = world.storage(Orbit),
        cameras = world.storage(Camera);
    const building = layout(SEED, POPULATION).find((p) => p.material === 1 && p.scale[1] > 25)!;
    orbit.pitch.set(camera, Math.atan(1 / Math.sqrt(2)));
    orbit.size.set(camera, 12);
    step(60);
    const t = world.storage(Transform).translation;
    orbit.pan.x.set(camera, building.translation[0] - t.x.get(camera));
    orbit.pan.y.set(camera, building.translation[1] - t.y.get(camera));
    orbit.pan.z.set(camera, building.translation[2] - t.z.get(camera));
    step(60);
    expect(t.x.get(camera)).toBeCloseTo(building.translation[0], 3);
    expect(t.y.get(camera)).toBeCloseTo(building.translation[1], 3);
    expect(t.z.get(camera)).toBeCloseTo(building.translation[2], 3);
    const negative = await frame();
    cameras.near.set(camera, 0.1);
    const positive = await frame();
    let retained = 0;
    for (let i = 0; i < positive.rgba.length; i += 4) {
        if (
            clear(positive.rgba[i], positive.rgba[i + 1], positive.rgba[i + 2]) &&
            !clear(negative.rgba[i], negative.rgba[i + 1], negative.rgba[i + 2])
        )
            retained++;
    }
    expect(retained).toBeGreaterThan(100);
});
