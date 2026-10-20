import { expect, test } from "bun:test";
import { Camera, World } from "@dylanebert/shallot";
import { LensDistortion, Orbit } from "@dylanebert/shallot/extras";
import { effects, Scene, ScenePlugin, setEffect } from "./scene";
import { setSetting, settings } from "./settings";

function scene(): World {
    const world = new World();
    ScenePlugin.initialize!(world);
    return world;
}

test("the scene starts with a single camera and no post-processing", () => {
    const world = scene();
    try {
        const camera = world.resource(Scene).camera;
        expect([...world.query([Camera])]).toEqual([camera]);
        expect(world.has(camera, Orbit)).toBe(true);
        for (const effect of effects) expect(world.has(camera, effect.component)).toBe(false);
    } finally {
        world.dispose();
    }
});

test("each control adds and removes only its own camera effect", () => {
    const world = scene();
    try {
        const camera = world.resource(Scene).camera;
        for (let index = 0; index < effects.length; index++) {
            setEffect(world, index, true);
            for (let other = 0; other < effects.length; other++)
                expect(world.has(camera, effects[other].component)).toBe(other <= index);
        }
        for (let index = 0; index < effects.length; index++) {
            setEffect(world, index, false);
            for (let other = 0; other < effects.length; other++)
                expect(world.has(camera, effects[other].component)).toBe(other > index);
        }
    } finally {
        world.dispose();
    }
});

for (const [effect, controls] of settings.entries()) {
    for (const [index, control] of controls.entries()) {
        test(`${effects[effect].name} ${control.label} writes its camera field`, () => {
            const world = scene();
            try {
                setEffect(world, effect, true);
                const camera = world.resource(Scene).camera;
                const value = (control.min + control.max) / 2;
                setSetting(world, effect, index, value);
                const actual =
                    effect === 0 && index > 0
                        ? world
                              .storage(LensDistortion)
                              .multiplier[index === 1 ? "x" : "y"].get(camera)
                        : world
                              .storage(effects[effect].component)
                              [control.field as "intensity"].get(camera);
                expect(actual).toBeCloseTo(value);
            } finally {
                world.dispose();
            }
        });
    }
}
