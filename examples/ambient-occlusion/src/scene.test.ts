import { expect, test } from "bun:test";
import {
    Diagnostic,
    DiagnosticsStore,
    ScreenSpaceAmbientOcclusion,
    World,
} from "@dylanebert/shallot";
import { Scene } from "./scene";
import { cost, setQuality, setSetting, settings } from "./settings";

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

test("cost sums only SSAO GPU spans, clears while off and reports unavailable timestamps", () => {
    const world = scene();
    try {
        const store = world.resource(DiagnosticsStore);
        expect(cost(world)).toBe("—");
        store.record("render/timestamp_query", 1);
        expect(cost(world)).toBe("—");
        store.add(new Diagnostic("render/standard:ssao-preprocess/elapsed_gpu"));
        expect(cost(world)).toBe("—");
        store.record("render/standard:ssao-preprocess/elapsed_gpu", 0.2);
        store.record("render/standard:ssao-evaluate/elapsed_gpu", 0.8);
        store.record("render/standard:ssao-evaluate/elapsed_cpu", 30);
        store.record("render/other/elapsed_gpu", 40);
        expect(cost(world)).toBe("1.00 ms");
        world.remove(world.resource(Scene).camera, ScreenSpaceAmbientOcclusion);
        expect(cost(world)).toBe("0.00 ms");
    } finally {
        world.dispose();
    }
});
