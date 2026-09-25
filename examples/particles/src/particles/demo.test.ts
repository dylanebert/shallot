import { resolve } from "node:path";
import { build, Compute, probeBuffer, Time } from "@dylanebert/shallot";
import { OrbitPlugin } from "@dylanebert/shallot/extras";
import { check } from "@dylanebert/shallot/harness/check";
import { SearPlugin } from "@dylanebert/shallot/standard/rendering";
import { PARTICLE_COUNT } from "./kernel";
import { ParticlesPlugin, particleState } from "./particles";

const SCENE = resolve(import.meta.dir, "../../public/scenes/particles.scene");

check(
    "particles move between submitted frames",
    {
        claim: "particles' actual scene dispatch changes GPU positions between submitted frames",
        size: "integration",
        requires: ["gpu"],
        subject: "examples/particles/src/particles/particles.ts",
    },
    async () => {
        const peerModule = "bun-webgpu";
        const peer = (await import(peerModule)) as { setupGlobals(): Promise<void> };
        await peer.setupGlobals();

        const app = await build({
            defaults: false,
            plugins: [OrbitPlugin, SearPlugin, ParticlesPlugin],
            scene: SCENE,
        });
        try {
            const device = Compute.device;
            const particles = particleState();
            if (!device || !particles) throw new Error("particles did not allocate its GPU buffer");

            app.state.step(Time.FIXED_DT);
            const first = await probeBuffer(device, particles.raw);
            app.state.step(Time.FIXED_DT);
            const second = await probeBuffer(device, particles.raw);
            const a = new Float32Array(first.bytes);
            const b = new Float32Array(second.bytes);
            let changed = 0;
            for (let i = 0; i < PARTICLE_COUNT; i++) {
                const lane = i * 8;
                if (
                    a[lane] !== b[lane] ||
                    a[lane + 1] !== b[lane + 1] ||
                    a[lane + 2] !== b[lane + 2]
                )
                    changed++;
            }
            if (changed === 0)
                throw new Error("no particle position changed between submitted frames");
        } finally {
            app.dispose();
        }
    },
);
