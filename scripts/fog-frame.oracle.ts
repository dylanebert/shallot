import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { shallot } from "@dylanebert/shallot/bun";
import { plugin } from "bun";

plugin(shallot({ root: import.meta.dir }));
const { createApp } = await import("@dylanebert/shallot");
const { attachTexture, captureTexture } = await import("../src/core/rendering");
const { setupGlobals } = await import("@dylanebert/shallot/webgpu");
const { Clock, CyclePlugin, CycleScene } = await import("../examples/day-night-cycle/src/cycle");
await setupGlobals();
const app = await createApp({ plugins: [CyclePlugin] });
const { world } = app;
const camera = world.resource(CycleScene).camera;
attachTexture(world, camera, { width: 320, height: 240 });
const directory = process.argv[2]!;
mkdirSync(directory, { recursive: true });
console.log(world.gpu.adapter);
for (const hour of [0, 6, 10, 12, 18, 22]) {
    world.resource(Clock).hour = hour;
    world.step(0);
    world.step(0);
    const rgba = (await captureTexture(world, camera)).rgba;
    const path = `${directory}/${hour}.rgba`;
    if (process.argv[3]) {
        const before = readFileSync(`${process.argv[3]}/${hour}.rgba`);
        const max = [0, 0, 0, 0];
        for (let i = 0; i < rgba.length; i++)
            max[i % 4] = Math.max(max[i % 4]!, Math.abs(rgba[i]! - before[i]!));
        console.log({ hour, max });
    }
    writeFileSync(path, rgba);
}
app.dispose();
process.exit(0);
