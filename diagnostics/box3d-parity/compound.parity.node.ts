import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp } from "@dylanebert/shallot";
import { StandardPhysicsPlugin } from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { compoundScenes } from "../../examples/physics/src/compound";
import { createScene } from "../../examples/physics/src/scenes";
import { physicsState } from "../../src/standard/physics/runtime";
import type { CompoundData } from "../../src/standard/physics/shapes/compound";
import { compoundRecord } from "./compound-records";

setDefaultTimeout(TIMEOUT.node);
for (const scene of Object.keys(compoundScenes) as (keyof typeof compoundScenes)[]) {
    test(`${scene} built compound children equal native geometry and reject a child mutation`, async () => {
        const expected = (await Bun.file(new URL(`../../examples/physics/tests/${scene}.compound.gold.txt`, import.meta.url)).text()).trimEnd().split("\n");
        const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin], setup(world) { createScene(world, scene); } });
        try {
            app.world.tick();
            const compounds = [...physicsState(app.world)!.geometryIdentityValues.values()].filter((v): v is CompoundData => "hulls" in v && "spheres" in v);
            expect(compounds.map(compoundRecord)).toEqual(expected);
            for (const data of compounds) {
                if (data.hulls.length) data.hulls[0]!.transform.p.x += 0.125;
                else if (data.spheres.length) data.spheres[0]!.sphere.radius += 0.125;
                else data.meshes[0]!.scale.x += 0.125;
            }
            expect(compounds.map(compoundRecord)).not.toEqual(expected);
        } finally { app.dispose(); }
    });
}
