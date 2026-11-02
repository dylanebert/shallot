import { expect, setDefaultTimeout, test } from "bun:test";
import { Camera, MeshInstance, MeshMaterial, Shape, Transform } from "@dylanebert/shallot";
import { Gizmo, Orbit } from "@dylanebert/shallot/extras";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { AppPlugin, Viewer } from "../src/app";
import { compoundChildren } from "../src/compound-presentation";
import { sceneSwitcher } from "../src/sample";
import { VillageState } from "../src/village";

setDefaultTimeout(TIMEOUT.node);
for (const scene of ["village", "benchmarkVillage"] as const)
    test(`${scene} retains shared child geometry for GPU culling across camera moves`, async () => {
        const app = await openGpuApp({ plugins: [AppPlugin] });
        try {
            const world = app.world;
            const select = sceneSwitcher(world, world.resource(Viewer));
            select(scene);
            world.tick();
            world.step(0);
            await world.frameFence;
            const camera = [...world.query([Camera])][0]!;
            const owner = [...world.query([Shape])][0]!;
            const children = compoundChildren(world, owner);
            expect(children).toHaveLength(scene === "village" ? 52500 : 11214);
            function check() {
                const expected = children;
                const mover = world.resource(VillageState).entity;
                const drawn = [...world.query([MeshInstance])].filter(
                    (eid) => eid !== mover && !world.has(eid, Shape),
                );
                expect(drawn).toHaveLength(expected.length);
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
                }
                return drawn;
            }
            const opening = check();
            expect(opening).toHaveLength(children.length);
            world.storage(Orbit).pan.set(camera, 3000, 0, 0);
            world.step(1 / 60);
            await world.frameFence;
            expect(check()).toEqual(opening);
            select("boxStack");
            expect([...world.query([Shape, MeshInstance])]).toHaveLength(41);
        } finally {
            app.dispose();
        }
    });
