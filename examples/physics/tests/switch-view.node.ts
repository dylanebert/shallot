import { expect, setDefaultTimeout, test } from "bun:test";
import { Body, Meshes, MeshInstance, Shape, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { LinesPlugin, OrbitPlugin, TextPlugin } from "@dylanebert/shallot/extras";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { EventSamplePlugin, SampleInputPlugin } from "../src/events";
import { Grab, GrabPlugin } from "../src/grab";
import { sceneSwitcher } from "../src/sample";
import { type Scene, scenes } from "../src/scenes";
import { createViewer, GrabFeedback } from "../src/view";

setDefaultTimeout(TIMEOUT.node);

test("scene presentation releases old meshes, systems and grab feedback in the running app", async () => {
    const app = await openGpuApp({
        plugins: [
            StandardPhysicsPlugin,
            EventSamplePlugin,
            SampleInputPlugin,
            TextPlugin,
            OrbitPlugin,
            LinesPlugin,
            GrabPlugin,
        ],
    });
    try {
        const world = app.world;
        const select = sceneSwitcher(world, createViewer(world));
        for (const scene of Object.keys(scenes) as Scene[]) {
            select(scene);
            const meshes = world.resource(Meshes);
            if (scene === "sensor") {
                const groundMesh = [...meshes.entries()].find(([, mesh]) => mesh.name === "Ground");
                expect(groundMesh).toBeDefined();
                const ground = [...world.query([MeshInstance])].filter(
                    (eid) => world.storage(MeshInstance).mesh.get(eid) === groundMesh![0],
                );
                expect(ground).toHaveLength(1);
                expect(world.has(ground[0]!, Body)).toBe(false);
                expect(world.has(ground[0]!, Shape)).toBe(false);
            }
            const freshCount = meshes.size;
            const freshEntities = [...world.query([MeshInstance])].length;
            select("contact");
            for (let i = 0; i < 60; i++) world.tick();
            world.step(0);
            const used = new Set(
                [...world.query([MeshInstance])].map((eid) =>
                    world.storage(MeshInstance).mesh.get(eid),
                ),
            );
            const old = [...meshes.entries()]
                .map(([handle]) => handle)
                .filter((handle) => used.has(handle));
            world.resource(Grab).outlined = 999;
            world.resource(GrabFeedback).marker = 999;
            select(scene);
            expect(meshes.size, scene).toBe(freshCount);
            expect([...world.query([MeshInstance])].length, scene).toBe(freshEntities);
            for (const handle of old) expect(meshes.has(handle)).toBe(false);
            expect(world.resource(GrabFeedback)).toEqual(GrabFeedback.create());
            expect(world.resource(Grab)).toEqual(Grab.create());
            world.step(0);
            expect(meshes.size, `${scene} after presentation`).toBe(freshCount);
        }
    } finally {
        app.dispose();
    }
});
