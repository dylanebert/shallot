import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { createApp } from "@dylanebert/shallot";
import * as physics from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { MeshSamplePlugin } from "../src/mesh";
import { createScene, scenes } from "../src/scenes";
import { drawingDigest } from "./digest";

setDefaultTimeout(TIMEOUT.node);
test("Creation Benchmark's one-frame triangle count and build work equal native without automatic stepping", async () => {
    const expected = (
        await Bun.file(new URL("./meshCreation.readouts.gold.txt", import.meta.url)).text()
    )
        .trim()
        .split("\n")[0]!;
    const app = await createApp({
        defaults: false,
        plugins: [MeshSamplePlugin],
        setup(world) {
            createScene(world, "meshCreation");
        },
    });
    const builds = spyOn(physics, "createPhysicsMesh");
    const releases = spyOn(physics, "destroyPhysicsMesh");
    try {
        expect(app.world.resource(physics.PhysicsWorldDefinition).enableAutomaticStep).toBe(false);
        const count = scenes.meshCreation.controls["triangle count"]!;
        expect(count.read!(app.world)).toBeNull();
        app.world.tick();
        expect(drawingDigest([[count.read!(app.world)!, builds.mock.calls.length]])).toBe(expected);
        expect(releases.mock.calls).toHaveLength(builds.mock.calls.length);
    } finally {
        builds.mockRestore();
        releases.mockRestore();
        await app.dispose();
    }
});
