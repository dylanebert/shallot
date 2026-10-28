import { expect, setDefaultTimeout, test } from "bun:test";
import { readFileSync } from "node:fs";
import { createApp, PhysicsMeshes, Shape, ShapeKind } from "../../src";
import { StandardPhysicsPlugin } from "../../src/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { createScene } from "../../examples/physics/src/scenes";
import { meshRecords } from "./mesh-records";
setDefaultTimeout(TIMEOUT.node);
for (const scene of ["voxel", "hollowBox", "gridMesh", "bigBoxMesh", "boxMesh"] as const) {
    test(`${scene} static mesh geometry equals native records`,async()=>{
        const expected = readFileSync(new URL(`../../examples/physics/tests/${scene}.mesh.gold.txt`,import.meta.url),"utf8").trimEnd().split("\n");
        const app = await createApp({defaults:false,plugins:[StandardPhysicsPlugin],setup(world){createScene(world,scene);}});
        try {
            app.world.tick();expect(meshRecords(app.world)).toEqual(expected);
            const eid = [...app.world.query([Shape])].find(eid=>app.world.get(eid,Shape)!.kind===ShapeKind.Mesh)!;
            const meshes = app.world.resource(PhysicsMeshes);
            const definition = structuredClone(meshes.get(app.world.get(eid,Shape)!.geometry)!);
            if ("grid" in definition) definition.grid.cellWidth += 0.1;
            else if ("vertices" in definition) definition.vertices[0] += 0.1;
            else throw new Error("unexpected mesh definition");
            app.world.storage(Shape).geometry.set(eid,meshes.add(definition));
            app.world.tick();expect(meshRecords(app.world)).not.toEqual(expected);
        }finally{app.dispose();}
    });
}
