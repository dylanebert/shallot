import { expect, test } from "bun:test";
import { ScreenSpaceAmbientOcclusion, World } from "@dylanebert/shallot";
import { Scene } from "./scene";
import { setQuality, setSetting, settings } from "./settings";

function scene(): World {
    const world = new World();
    const camera = world.create();
    world.resource(Scene).camera = camera;
    world.add(camera, ScreenSpaceAmbientOcclusion);
    return world;
}

for (const [index, control] of settings.entries()) {
    test(`${control.label} writes its camera field`, () => {
        const world = scene();
        try {
            const camera = world.resource(Scene).camera;
            const value = (control.min + control.max) / 2;
            setSetting(world, index, value);
            const storage = world.storage(ScreenSpaceAmbientOcclusion);
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
