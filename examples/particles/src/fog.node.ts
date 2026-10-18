import { afterAll, beforeAll, describe, expect, setDefaultTimeout, test } from "bun:test";
import { type createApp, Exposure, GlobalAmbientLight } from "@dylanebert/shallot";
import { NoFog } from "@dylanebert/shallot/extras";
import { attachTexture, captureTexture } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { Motes, MotesPlugin } from "./motes";
import { Room } from "./room";

setDefaultTimeout(TIMEOUT.node);
const WIDTH = 1280;
const LEFT = 450;
const RIGHT = 530;
function linear(value: number) {
    const x = value / 255;
    return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
}
function luminance(rgba: Uint8ClampedArray, x: number, y: number) {
    const at = (y * WIDTH + x) * 4;
    return (
        0.2126 * linear(rgba[at]!) + 0.7152 * linear(rgba[at + 1]!) + 0.0722 * linear(rgba[at + 2]!)
    );
}
function profile(rgba: Uint8ClampedArray) {
    const values = new Float64Array(RIGHT - LEFT);
    for (let x = LEFT; x < RIGHT; x++) {
        for (let y = 431; y <= 435; y++) values[x - LEFT] += luminance(rgba, x, y) / 5;
    }
    return values;
}
// The room's sun through its window, interreflected by its walls (ρΦ / πA(1 − ρ)), lights it to
// about 9.2 cd/m²; the example's brighter ambient hides the shaft edge's temporal swing.
const BALANCED_AMBIENT = 9.2;
for (const ev100 of [5, 6]) {
    describe(`at the balanced ambient and EV${ev100}`, () => {
        let app: Awaited<ReturnType<typeof createApp>>;
        let profileError = 0;
        let patchSwing = 0;
        beforeAll(async () => {
            app = await openGpuApp({ plugins: [MotesPlugin] });
            const { world } = app;
            console.log("shaft adapter", world.gpu.adapter);
            const camera = world.resource(Room).camera;
            world.resource(Motes).visible = false;
            world.resource(GlobalAmbientLight).brightness = BALANCED_AMBIENT;
            world.storage(Exposure).ev100.set(camera, ev100);
            attachTexture(world, camera, { width: WIDTH, height: 720 });
            world.step(1 / 60);
            world.step(1 / 60);
            for (let i = 0; i < 64; i++) {
                world.step(0);
                await world.gpu.sync();
            }
            const profiles: Float64Array[] = [];
            const patches: number[] = [];
            for (let i = 0; i < 16; i++) {
                world.step(0);
                const { rgba } = await captureTexture(world, camera);
                profiles.push(profile(rgba));
                let patch = 0;
                for (let y = 431; y <= 435; y++) {
                    for (let x = 489; x <= 493; x++) patch += luminance(rgba, x, y) / 25;
                }
                patches.push(patch);
            }
            const reference = new Float64Array(RIGHT - LEFT);
            for (let i = 0; i < 16; i++) {
                // Opt-out discards history. A three-frame stride visits all 16 Halton phases.
                world.add(camera, NoFog);
                world.step(0);
                world.step(0);
                world.remove(camera, NoFog);
                world.step(0);
                const current = profile((await captureTexture(world, camera)).rgba);
                for (let x = 0; x < reference.length; x++) reference[x] += current[x]! / 16;
            }
            for (const current of profiles) {
                for (let x = 0; x < reference.length; x++)
                    profileError = Math.max(profileError, Math.abs(current[x]! - reference[x]!));
            }
            patchSwing = Math.max(...patches) - Math.min(...patches);
            console.log("shaft temporal error", { ev100, profileError, patchSwing, patches });
        });
        afterAll(() => app?.dispose());

        test("every converged shaft-edge profile agrees with the equal 16-phase linear average", () => {
            expect(profileError).toBeLessThan(0.005);
        });

        test("a still shaft-edge patch stays calm across all 16 consecutive phases", () => {
            expect(patchSwing).toBeLessThan(0.005);
        });
    });
}
