import { expect, test } from "bun:test";
import { Camera, ScreenSpaceAmbientOcclusion, World } from "@dylanebert/shallot";
import { AO_LEVELS, addAmbientOcclusion } from "./ssao";

test("capture quality selection applies Bevy's presets to every camera and leaves Off unrequested", () => {
    for (const [index, level] of AO_LEVELS.entries()) {
        const world = new World();
        world.registry.register(Camera);
        world.registry.register(ScreenSpaceAmbientOcclusion);
        const cameras = [world.create(), world.create()];
        for (const camera of cameras) world.add(camera, Camera);
        addAmbientOcclusion(world, level);
        for (const camera of cameras) {
            expect(world.has(camera, ScreenSpaceAmbientOcclusion)).toBe(index !== 0);
            if (index === 0) continue;
            const fields = world.storage(ScreenSpaceAmbientOcclusion);
            expect(fields.qualityLevel.get(camera)).toBe(index - 1);
            expect(fields.constantObjectThickness.get(camera)).toBe(0.25);
            expect(fields.sliceCount.get(camera)).toBe(3);
            expect(fields.samplesPerSliceSide.get(camera)).toBe(3);
        }
        world.dispose();
    }
});

test("an unknown quality is refused instead of silently capturing Off", () => {
    const world = new World();
    expect(() => addAmbientOcclusion(world, "low")).toThrow("Unknown SSAO quality");
    world.dispose();
});
