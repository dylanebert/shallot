import { expect, test } from "bun:test";
import { ColorGrading, World } from "@dylanebert/shallot";
import { Scene } from "./scene";
import { setSetting, settings } from "./settings";

for (const [index, control] of settings.entries()) {
    test(`${control.group} ${control.label} writes its camera field and restores identity`, () => {
        const world = new World();
        try {
            const camera = world.create();
            world.resource(Scene).camera = camera;
            world.add(camera, ColorGrading);
            const storage = world.storage(ColorGrading);
            const read = () =>
                control.axis
                    ? storage[control.field as "gain"][control.axis].get(camera)
                    : storage[control.field as "exposure"].get(camera);
            setSetting(world, index, control.max);
            expect(read()).toBeCloseTo(control.max);
            setSetting(world, index, control.value);
            expect(read()).toBe(control.value);
            expect(control.min + control.max).toBeCloseTo(control.value * 2);
        } finally {
            world.dispose();
        }
    });
}
