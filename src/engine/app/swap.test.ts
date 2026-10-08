import { expect, test } from "bun:test";
import { World } from "../ecs";
import { type Plugin, swapPlugins } from "./index";

test("a recovery factory failure is a named rebuild result, like an initialize failure", async () => {
    const world = new World();
    const old: Plugin = {
        name: "RecoveryOwner",
        recovery: () => ({ snapshot: () => undefined, restore() {} }),
    };
    const next: Plugin = {
        name: old.name,
        recovery() {
            throw "unsupported image";
        },
    };
    try {
        world.registerRecovery(
            old.name,
            typeof old.recovery === "function" ? old.recovery(world) : old.recovery,
        );
        expect(await swapPlugins(world, [old], [next])).toEqual({
            ok: false,
            reason: "RecoveryOwner: recovery threw — unsupported image",
        });
    } finally {
        world.dispose();
    }
});

for (const scheduling of [{ boundary: "before" as const }, { terminal: true }]) {
    test(`changing ${Object.keys(scheduling)[0]} refuses a swap before mutating the live system`, async () => {
        const world = new World();
        const update = () => {};
        const old: Plugin = { name: "SchedulingOwner", systems: [{ group: "simulation", update }] };
        const next: Plugin = {
            name: old.name,
            systems: [
                {
                    group: "simulation",
                    ...scheduling,
                    update() {
                        throw new Error("must not swap");
                    },
                },
            ],
        };
        try {
            world.addSystem(old.systems![0], old.name);
            expect(await swapPlugins(world, [old], [next])).toEqual({
                ok: false,
                reason: "SchedulingOwner: system 0 scheduling changed",
            });
            expect(old.systems![0].update).toBe(update);
        } finally {
            world.dispose();
        }
    });
}
