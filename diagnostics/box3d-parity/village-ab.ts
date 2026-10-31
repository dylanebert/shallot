// Village's composed sample, with the hash and phase records ab.ts consumes.
import { createApp, hashPhysics } from "../../src";
import { PhysicsWorldDefinition } from "../../src/standard/physics/world-definition";
import { physicsState } from "../../src/standard/physics/runtime";
import { worldProfile } from "../../src/standard/physics/world/world";
import { PROFILE_FIELDS } from "../../src/standard/physics/world/profile";
import { VillagePlugin } from "../../examples/physics/src/village";
import { createScene } from "../../examples/physics/src/scenes";

const [, threadArg, stepArg] = process.argv.slice(2);
const from = Number(process.env.PROFILE);
const steps = Number(stepArg);
const threads = Number(threadArg);
const app = await createApp({
    defaults: false,
    plugins: [VillagePlugin],
    setup(world) {
        world.resource(PhysicsWorldDefinition).workerCount = threads === 1 ? 0 : threads;
        world.resource(PhysicsWorldDefinition).maximumMemory = 4 * 1024 ** 3;
    },
});
try {
    createScene(app.world, "village");
    const lines: string[] = [];
    const hashSteps = process.env.HASH_STEPS !== "0";
    const state = physicsState(app.world)!;
    const loopStart = performance.now();
    let before = 0, window = 0, hashing = 0;
    for (let i = 0; i < steps; ++i) {
        const start = performance.now();
        app.world.tick();
        const hashStart = performance.now();
        if (i < from) before += hashStart - start;
        else window += hashStart - start;
        if (hashSteps) {
            lines.push(`${i} 0x${hashPhysics(app.world).toString(16).padStart(16, "0")}`);
            hashing += performance.now() - hashStart;
        }
        if (i >= from) {
            const profile = worldProfile(state);
            lines.push(`F ${i} ${PROFILE_FIELDS.map(k => profile[k].toFixed(4)).join(" ")}`);
        }
    }
    lines.push(`M ${JSON.stringify({ setup: loopStart, before, hashing, window, other: performance.now() - loopStart - before - hashing - window })}`);
    console.log(lines.join("\n"));
} finally {
    await app.dispose();
}
