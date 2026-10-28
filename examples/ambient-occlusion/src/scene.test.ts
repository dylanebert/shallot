import { expect, test } from "bun:test";
import { ScreenSpaceAmbientOcclusion, World } from "@dylanebert/shallot";
import { Scene, setOcclusion } from "./scene";
import { setQuality, setSetting, settings } from "./settings";

function scene(): World {
    const world = new World();
    world.registry.register(ScreenSpaceAmbientOcclusion);
    const camera = world.create();
    world.resource(Scene).camera = camera;
    setOcclusion(world, true);
    return world;
}

for (const [index, control] of settings.entries()) {
    test(`${control.label} starts at its camera field and writes it`, () => {
        const world = scene();
        try {
            const camera = world.resource(Scene).camera;
            const storage = world.storage(ScreenSpaceAmbientOcclusion);
            expect(storage[control.field].get(camera)).toBe(Math.fround(control.value));
            const value = (control.min + control.max) / 2;
            setSetting(world, index, value);
            expect(storage[control.field].get(camera)).toBeCloseTo(value);
        } finally {
            world.dispose();
        }
    });
}

test("quality changes the camera level", () => {
    const world = scene();
    try {
        setQuality(world, 1);
        expect(
            world
                .storage(ScreenSpaceAmbientOcclusion)
                .qualityLevel.get(world.resource(Scene).camera),
        ).toBe(1);
    } finally {
        world.dispose();
    }
});
