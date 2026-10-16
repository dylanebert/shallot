import { expect, setDefaultTimeout, test } from "bun:test";
import { Sky, SkyPlugin } from "@dylanebert/shallot/extras";
import { MeshInstance } from "@dylanebert/shallot/mesh";
import {
    AmbientLight,
    attachTexture,
    Camera,
    captureTexture,
    DirectionalLight,
    Exposure,
    SunDisk,
    Tonemapping,
    TonemappingMethod,
} from "@dylanebert/shallot/rendering";
import {
    Backgrounds,
    CameraBackground,
    Materials,
    MeshMaterial,
    StandardMaterial,
    StandardRenderer,
} from "@dylanebert/shallot/standard/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Transform } from "@dylanebert/shallot/transform";
import { gpuApps } from "../../../scripts/gpu.fixture";

setDefaultTimeout(TIMEOUT.node);
const subjects = gpuApps(import.meta.path, [{ plugins: [SkyPlugin] }]);

function linear(byte: number): number {
    const x = byte / 255;
    return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
}

test("one EV dims the sky, SunDisk disk and glow, and a lit surface by half", async () => {
    const { world } = subjects()[0]!;
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    world.add(camera, AmbientLight, { brightness: 500 });
    world.add(camera, Exposure, { ev100: 9.7 });
    world.add(camera, CameraBackground, { name: world.resource(Backgrounds).id("sky") });
    attachTexture(world, camera, { width: 17, height: 17 });
    const sky = world.create();
    world.add(sky, Sky, { zenith: 0x808080, horizon: 0x808080, cloudCoverage: 0, hazeDensity: 0 });
    const light = world.create();
    world.add(light, Transform, { rotation: [0, 1, 0, 0] });
    world.add(light, DirectionalLight, { illuminance: 0 });
    world.add(light, SunDisk, { intensity: 0.2, glow: 0 });
    const mesh = world.create();
    world.add(mesh, Transform);
    world.add(mesh, MeshInstance);
    world.add(
        mesh,
        MeshMaterial,
        world.resource(Materials).add(StandardMaterial({ baseColor: [0.5, 0.5, 0.5, 1] })),
    );
    async function value(ev100: number) {
        world.storage(Exposure).ev100.set(camera, ev100);
        world.step(0);
        world.step(0);
        const { rgba } = await captureTexture(world, camera);
        return linear(rgba[(8 * 17 + 8) * 4]!);
    }
    for (const subject of ["surface", "disk", "glow", "sky"]) {
        if (subject === "disk") {
            world.remove(mesh, MeshInstance);
            world.storage(Sky).zenith.set(sky, 0);
            world.storage(Sky).horizon.set(sky, 0);
        }
        if (subject === "glow") {
            world.storage(SunDisk).intensity.set(light, 0);
            world.storage(SunDisk).glow.set(light, 0.2);
        }
        if (subject === "sky") {
            world.remove(light, SunDisk);
            world.storage(Sky).zenith.set(sky, 0x808080);
            world.storage(Sky).horizon.set(sky, 0x808080);
        }
        const before = await value(9.7);
        const after = await value(10.7);
        expect(before, subject).toBeGreaterThan(0.02);
        expect(Math.abs(after / before - 0.5), subject).toBeLessThan(0.025);
    }
});
