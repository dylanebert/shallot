import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { World } from "../../engine";
import { init, shutdown } from "./kernel/kernel";
import { StandardPhysicsPlugin } from "./runtime";

setDefaultTimeout(TIMEOUT.node);

for (const [count, artifact] of [
    [0, "single"],
    [1, "shared"],
] as const) {
    test(`physics startup observes the initialized ${artifact} artifact`, async () => {
        const world = new World();
        const previous = Object.getOwnPropertyDescriptor(globalThis, "window");
        const browser = new EventTarget();
        const observations: unknown[] = [];
        browser.addEventListener("shallot:startup", (event) =>
            observations.push((event as CustomEvent).detail),
        );
        try {
            await init(world, { threads: count });
            await StandardPhysicsPlugin.initialize!(world);
            Object.defineProperty(globalThis, "window", { configurable: true, value: browser });
            await StandardPhysicsPlugin.warm!(world);
            expect(observations).toEqual([{ kind: "physics", threads: 1, artifact }]);
        } finally {
            if (previous) Object.defineProperty(globalThis, "window", previous);
            else Reflect.deleteProperty(globalThis, "window");
            StandardPhysicsPlugin.dispose!(world);
            await shutdown(world);
            world.dispose();
        }
    });
}
