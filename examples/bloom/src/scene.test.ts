import { expect, test } from "bun:test";
import { World } from "@dylanebert/shallot";
import { Bloom } from "@dylanebert/shallot/bloom";
import { Scene } from "./scene";
import { setMode, setSetting, settings } from "./settings";

function scene(): World {
    const world = new World();
    const camera = world.create();
    world.resource(Scene).camera = camera;
    world.add(camera, Bloom);
    return world;
}

for (const [index, control] of settings.entries()) {
    test(`${control.label} writes its camera field`, () => {
        const world = scene();
        try {
            const camera = world.resource(Scene).camera;
            const value = (control.min + control.max) / 2;
            setSetting(world, index, value);
            const storage = world.storage(Bloom);
            expect(
                control.field === "scale"
                    ? storage.scale.x.get(camera)
                    : storage[control.field].get(camera),
            ).toBeCloseTo(value);
        } finally {
            world.dispose();
        }
    });
}

test("mode changes the camera composite", () => {
    const world = scene();
    try {
        setMode(world, 1);
        expect(world.storage(Bloom).compositeMode.get(world.resource(Scene).camera)).toBe(1);
    } finally {
        world.dispose();
    }
});
