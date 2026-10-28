import { expect, setDefaultTimeout, test } from "bun:test";
import { Camera, type Plugin } from "@dylanebert/shallot";
import { PhysicsWorldDefinition } from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { MotesPlugin as ParticlesMotes } from "../examples/particles/src/motes";
import { MotesPlugin as FogMotes } from "../examples/volumetric-fog/src/motes";

setDefaultTimeout(TIMEOUT.node);

for (const [family, MotesPlugin, label] of [
    ["particles", ParticlesMotes, "particles:draw"],
    ["volumetric-fog", FogMotes, "dust:draw"],
] as const) {
    for (const samples of [0, 1, 4]) {
        test(`${family}: initial sample count ${samples} compiles only the reachable motes variant`, async () => {
            const counts: number[] = [];
            const control: Plugin = {
                name: "MotesPreparationControl",
                dependencies: [MotesPlugin],
                initialize(world) {
                    for (const eid of world.query([Camera])) {
                        if (!samples) world.remove(eid, Camera);
                        else world.storage(Camera).antialias.set(eid, samples === 1 ? 0 : 1);
                    }
                    const device = world.gpu.device;
                    const sync = device.createRenderPipeline.bind(device);
                    const async = device.createRenderPipelineAsync.bind(device);
                    device.createRenderPipeline = (descriptor) => {
                        if (descriptor.label === label)
                            counts.push(descriptor.multisample?.count ?? 1);
                        return sync(descriptor);
                    };
                    device.createRenderPipelineAsync = (descriptor) => {
                        if (descriptor.label === label)
                            counts.push(descriptor.multisample?.count ?? 1);
                        return async(descriptor);
                    };
                },
            };
            const app = await openGpuApp({
                pixelRatio: 1,
                setup(world) {
                    world.resource(PhysicsWorldDefinition).workerCount = 1;
                },
                plugins: [control],
            });
            try {
                expect(counts).toEqual(samples ? [samples] : []);
            } finally {
                app.dispose();
            }
        });
    }
}
