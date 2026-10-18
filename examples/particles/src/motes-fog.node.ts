import { expect, setDefaultTimeout, test } from "bun:test";
import { Exposure, Transform, VolumetricLight } from "@dylanebert/shallot";
import { Fog, NoFog } from "@dylanebert/shallot/extras";
import {
    attachTexture,
    captureTexture,
    Tonemapping,
    TonemappingMethod,
} from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { MOTE_COUNT, Motes, MotesPlugin } from "./motes";
import { Player } from "./player";
import { Room } from "./room";

setDefaultTimeout(TIMEOUT.node);
const decode = (x: number) => {
    const v = x / 255;
    return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
};

test("a mote takes local light exposed once, is zero with NoFog, and takes its own transmittance", async () => {
    const app = await openGpuApp({ plugins: [MotesPlugin] });
    try {
        const { world } = app;
        const camera = world.resource(Room).camera;
        world.remove(world.resource(Room).player, Player);
        attachTexture(world, camera, { width: 1024, height: 1024 });
        world.storage(Transform).translation.set(camera, 0, 0.7, 2, 0);
        world.tick();
        world.add(camera, Tonemapping, { method: TonemappingMethod.None });
        world.storage(Exposure).ev100.set(camera, 12);
        const fog = world.only([Fog]);
        world.storage(Fog).color.set(fog, 0);
        world.storage(Fog).absorption.set(fog, 1);
        world.storage(Fog).scattering.set(fog, 0);
        const motes = world.resource(Motes);
        const positions = new Float32Array(MOTE_COUNT * 4);
        for (let i = 0; i < MOTE_COUNT; i++) positions[i * 4] = 100;
        positions.set([0, 0.7, 0, 0]);
        world.gpu.device.queue.writeBuffer(motes.positions!, 0, positions);
        let previousDensity: number | undefined;
        const frame = async (density: number, visible: boolean, settle = 1) => {
            if (density !== previousDensity) settle = 32;
            previousDensity = density;
            world.storage(Fog).density.set(fog, density);
            motes.visible = visible;
            for (let i = 0; i < settle; i++) world.step(0);
            return (await captureTexture(world, camera)).rgba;
        };
        const clearBase = await frame(0.00001, false);
        const clearMote = await frame(0.00001, true);
        let at = 0,
            added = 0;
        for (let i = 0; i < clearMote.length; i += 4) {
            const delta = decode(clearMote[i]!) - decode(clearBase[i]!);
            if (delta > added) {
                at = i;
                added = delta;
            }
        }
        expect(added).toBeGreaterThan(0.01);
        const fogBase = await frame(0.2, false);
        const fogMote = await frame(0.2, true);
        const ratio = (decode(fogMote[at]!) - decode(fogBase[at]!)) / added;
        const nearT = Math.exp(-0.2 * (2 - 0.1));
        const wallT = Math.exp(-0.2 * (5.3 - 0.1));
        world.storage(Exposure).ev100.set(camera, 13);
        const halfBase = await frame(0.00001, false);
        const halfMote = await frame(0.00001, true);
        const halfAdded = decode(halfMote[at]!) - decode(halfBase[at]!);
        const exposedOnce = Math.abs(halfAdded / added - 0.5) < 0.05;
        world.add(camera, NoFog);
        const noFogBase = await frame(0.00001, false);
        const noFogMote = await frame(0.00001, true);
        const zeroWithNoFog = noFogMote[at] === noFogBase[at];
        world.remove(camera, NoFog);
        for (const sun of world.query([VolumetricLight])) world.remove(sun, VolumetricLight);
        const darkBase = await frame(0.00001, false, 32);
        const darkMote = await frame(0.00001, true);
        const followsLocalLight = decode(darkMote[at]!) - decode(darkBase[at]!) < added * 0.03;
        expect({ exposedOnce, zeroWithNoFog, followsLocalLight }).toEqual({
            exposedOnce: true,
            zeroWithNoFog: true,
            followsLocalLight: true,
        });
        expect(Math.abs(ratio - nearT)).toBeLessThan(0.04);
        expect(Math.abs(ratio - wallT)).toBeGreaterThan(0.2);
    } finally {
        app.dispose();
    }
});
