import { expect, test } from "bun:test";
import { Camera, World } from "@dylanebert/shallot";
import { effects, Scene, ScenePlugin, setEffect } from "./scene";

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
