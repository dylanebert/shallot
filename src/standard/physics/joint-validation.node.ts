import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { type Component, createApp, Time } from "@dylanebert/shallot";
import * as core from "@dylanebert/shallot/physics";
import { physicsWorld, StandardPhysicsPlugin } from "@dylanebert/shallot/standard/physics";
import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);
const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();

test("values asserted against by Box3D create warn once and skip, while positive infinite weld hertz is accepted", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const warning = spyOn(console, "warn").mockImplementation(() => {});
    const world = app.world;
    try {
        const a = world.create();
        const b = world.create();
        world.add(a, core.Body);
        world.add(b, core.Body);
        world.step(Time.FIXED_DT);
        const cases = [
            ["Filter", { localAnchorA: [Infinity, 0, 0, 0] }, "localFrameA"],
            ["Motor", { localRotationB: [0, 0, 0, 0] }, "localFrameB"],
            ["Distance", { length: 0 }, "length"],
            ["Distance", { length: Infinity }, "length"],
            [
                "Distance",
                { lowerSpringForce: 2, upperSpringForce: 1 },
                "lowerSpringForce/upperSpringForce",
            ],
            ["Parallel", { hertz: -1 }, "hertz"],
            ["Parallel", { dampingRatio: Infinity }, "dampingRatio"],
            ["Parallel", { maxTorque: NaN }, "maxTorque"],
            [
                "Prismatic",
                { lowerTranslation: 1, upperTranslation: 0 },
                "lowerTranslation/upperTranslation",
            ],
            ["Spherical", { coneAngle: Math.PI }, "coneAngle"],
            ["Spherical", { targetRotation: [0, 0, 0, 1.00001] }, "targetRotation"],
            ["Weld", { angularHertz: NaN }, "angularHertz"],
            ["Weld", { angularDampingRatio: -1 }, "angularDampingRatio"],
            ["Weld", { linearHertz: -1 }, "linearHertz"],
            ["Weld", { linearDampingRatio: -1 }, "linearDampingRatio"],
            [
                "Wheel",
                { lowerSuspensionLimit: 1, upperSuspensionLimit: 0 },
                "lowerSuspensionLimit/upperSuspensionLimit",
            ],
        ] as const;
        for (const [kind, values, field] of cases) {
            const component = (core as unknown as Record<string, Component>)[`${kind}Joint`];
            const eid = world.create();
            world.add(eid, component, { a, b, ...values });
            warning.mockClear();
            world.step(Time.FIXED_DT);
            world.step(Time.FIXED_DT);
            expect(warning).toHaveBeenCalledTimes(1);
            expect(String(warning.mock.calls[0]![0])).toContain(`${kind}Joint ${eid}`);
            expect(String(warning.mock.calls[0]![0])).toContain(field);
            expect(physicsWorld(world)!.state.joints.filter((j) => j.jointId >= 0)).toHaveLength(0);
            world.destroy(eid);
            world.step(Time.FIXED_DT);
        }
        const eid = world.create();
        world.add(eid, core.WeldJoint, { a, b, angularHertz: Infinity });
        warning.mockClear();
        world.step(Time.FIXED_DT);
        expect(warning).not.toHaveBeenCalled();
        expect(physicsWorld(world)!.state.joints.filter((j) => j.jointId >= 0)).toHaveLength(1);
        world.storage(core.WeldJoint).angularHertz.set(eid, NaN);
        world.step(Time.FIXED_DT);
        expect(warning).toHaveBeenCalledTimes(1);
        expect(physicsWorld(world)!.state.joints.filter((j) => j.jointId >= 0)).toHaveLength(0);
    } finally {
        warning.mockRestore();
        app.dispose();
    }
});
