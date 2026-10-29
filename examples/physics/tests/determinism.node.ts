import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { DeterminismPlugin, DeterminismState } from "../src/determinism";
import { querySpawnDrawing } from "../src/determinism-presentation";
import { createScene } from "../src/scenes";

setDefaultTimeout(TIMEOUT.node);
test("Query Spawn draws its ray, overlap bounds and stopped sphere in scene ink", async () => {
    const app = await createApp({ defaults: false, plugins: [DeterminismPlugin] });
    try {
        createScene(app.world, "querySpawn");
        expect(querySpawnDrawing(app.world)).toEqual([]);
        app.world.tick();
        const s = app.world.resource(DeterminismState),
            lines = querySpawnDrawing(app.world);
        expect(lines[0]).toEqual({ a: s.origin, b: s.rayPoint });
        expect(lines.slice(1, 13)).toHaveLength(12);
        for (const { a, b } of lines.slice(1, 13)) {
            for (const p of [a, b])
                for (let axis = 0; axis < 3; axis++)
                    expect([s.lower[axis], s.upper[axis]]).toContain(p[axis]);
            expect(a.filter((v, axis) => v !== b[axis])).toHaveLength(1);
        }
        expect(lines.length).toBeGreaterThan(13);
    } finally {
        app.dispose();
    }
});
