import { expect, setDefaultTimeout, test } from "bun:test";
import { Camera, MeshInstance, MeshMaterial, qRotate, Shape, Transform } from "@dylanebert/shallot";
import { Gizmo, Orbit } from "@dylanebert/shallot/extras";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { AppPlugin, Viewer } from "../src/app";
import { compoundChildren, overlaps } from "../src/compound-presentation";
import { drawingBounds } from "../src/draw-set";
import { sceneSwitcher } from "../src/sample";
import { VillageState } from "../src/village";

setDefaultTimeout(TIMEOUT.node);
test("Village shares child geometry and draws only children overlapping Box3D's camera bounds", async () => {
    const app = await openGpuApp({ plugins: [AppPlugin] });
    try {
        const world = app.world;
        const select = sceneSwitcher(world, world.resource(Viewer));
        select("village");
        world.tick();
        world.step(0);
        await world.frameFence;
        const camera = [...world.query([Camera])][0]!;
        const owner = [...world.query([Shape])][0]!;
        const children = compoundChildren(world, owner);
        expect(children).toHaveLength(52500);
        function check() {
            const pose = world.get(camera, Transform)!;
            const bounds = drawingBounds(
                pose.translation,
                qRotate(...pose.rotation, 0, 0, -1),
                1,
                world.storage(Camera).fov.get(camera),
            );
            const expected = children.filter((child) => overlaps(child.bounds, bounds));
            const mover = world.resource(VillageState).entity;
            const drawn = [...world.query([MeshInstance])].filter((eid) => eid !== mover);
            expect(drawn).toHaveLength(expected.length);
            expect(expected.length).toBeLessThan(children.length);
            expect(world.has(owner, MeshInstance)).toBe(false);
            const meshes = new Set(drawn.map((eid) => world.get(eid, MeshInstance)!.mesh));
            expect(meshes.size).toBe(new Set(expected.map((child) => child.key)).size);
            if (expected.length) expect(meshes.size).toBeLessThan(expected.length / 5);
            const positions = new Map(
                expected.map((child) => [child.translation.map(Math.fround).join(","), child]),
            );
            for (const eid of drawn) {
                expect(world.has(eid, MeshMaterial)).toBe(true);
                expect(world.has(eid, Gizmo)).toBe(true);
                const child = positions.get(world.get(eid, Transform)!.translation.join(","));
                expect(child).toBeDefined();
                expect(overlaps(child!.bounds, bounds)).toBe(true);
            }
            return expected.length;
        }
        const opening = check();
        world.storage(Orbit).pan.set(camera, 3000, 0, 0);
        world.step(1 / 60);
        await world.frameFence;
        expect(check()).toBe(0);
        console.log(`Village opening draw set: ${opening} / ${children.length} children`);
        select("boxStack");
        expect([...world.query([Shape, MeshInstance])]).toHaveLength(41);
    } finally {
        app.dispose();
    }
});
