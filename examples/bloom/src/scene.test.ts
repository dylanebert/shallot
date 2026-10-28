import { expect, test } from "bun:test";
import { World } from "@dylanebert/shallot";
import { Bloom } from "@dylanebert/shallot/bloom";
import { Scene, setBloom } from "./scene";
import { setMode, setSetting, settings } from "./settings";

function scene(): World {
    const world = new World();
    world.registry.register(Bloom);
    const camera = world.create();
    world.resource(Scene).camera = camera;
    setBloom(world, true);
    return world;
}

for (const [index, control] of settings.entries()) {
    test(`${control.label} starts at its camera field and writes it`, () => {
        const world = scene();
        try {
            const camera = world.resource(Scene).camera;
            const storage = world.storage(Bloom);
            const read = () =>
                control.field === "scale"
                    ? storage.scale.x.get(camera)
                    : storage[control.field].get(camera);
            expect(read()).toBe(Math.fround(control.value));
            const value = (control.min + control.max) / 2;
            setSetting(world, index, value);
            expect(read()).toBeCloseTo(value);
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
