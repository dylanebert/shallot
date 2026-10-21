import { expect, test } from "bun:test";
import { ColorGrading, Tonemapping, World } from "@dylanebert/shallot";
import { Scene } from "./scene";
import { methods, setMethod } from "./settings";

for (const [name, method] of methods) {
    test(`${name} applies Bevy's basic scene recommendation`, () => {
        const world = new World();
        try {
            const camera = world.create();
            world.resource(Scene).camera = camera;
            for (const [, previous] of methods) {
                setMethod(world, previous);
                setMethod(world, method);
                expect(world.storage(Tonemapping).method.get(camera)).toBe(method);
                const grade = world.storage(ColorGrading);
                expect(grade.exposure.get(camera)).toBeCloseTo(
                    name.startsWith("Reinhard")
                        ? 0.5
                        : name === "AcesFitted"
                          ? 0.35
                          : name === "AgX"
                            ? -0.2
                            : 0,
                );
                expect(grade.postSaturation.get(camera)).toBeCloseTo(name === "AgX" ? 1.1 : 1);
                for (const axis of ["x", "y", "z"] as const)
                    expect(grade.saturation[axis].get(camera)).toBeCloseTo(
                        name === "AgX" ? 1.1 : 1,
                    );
            }
        } finally {
            world.dispose();
        }
    });
}
