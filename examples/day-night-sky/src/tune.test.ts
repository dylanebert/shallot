import { resolve } from "node:path";
import { build, DirectionalLight, type Plugin, Time } from "@dylanebert/shallot";
import { OrbitPlugin, Sky } from "@dylanebert/shallot/extras";
import { check } from "@dylanebert/shallot/harness/check";
import SkyTune from "./tune";

const SCENE = resolve(import.meta.dir, "../public/scenes/day-night-sky.scene");

// The renderer owns the full visual composition; these scene facts are its CPU-readable contract.
const SceneFacts = {
    name: "SceneFacts",
    components: { DirectionalLight, Sky },
    traits: {
        DirectionalLight: { singleton: true },
        Sky: { singleton: true },
    },
} satisfies Plugin;

async function sampleCycle() {
    const app = await build({
        defaults: false,
        plugins: [OrbitPlugin, SceneFacts, SkyTune],
        scene: SCENE,
    });
    try {
        const state = app.state;
        const sun = state.only([DirectionalLight]);
        if (sun < 0) throw new Error("actual scene has no directional light");
        state.step(0);
        const start = [
            DirectionalLight.direction.x.get(sun),
            DirectionalLight.direction.y.get(sun),
            DirectionalLight.direction.z.get(sun),
        ];
        for (let i = 0; i < 30; i++) state.step(Time.FIXED_DT);
        return {
            elapsed: state.time.elapsed,
            start,
            end: [
                DirectionalLight.direction.x.get(sun),
                DirectionalLight.direction.y.get(sun),
                DirectionalLight.direction.z.get(sun),
            ],
        };
    } finally {
        app.dispose();
    }
}

check(
    "day-night-sky light follows elapsed across a rebuild",
    {
        claim: "day-night-sky sweeps the actual scene directional light from elapsed time and reproduces it at the same elapsed after a rebuild",
    },
    async () => {
        const first = await sampleCycle();
        if (first.end.every((value, i) => value === first.start[i]))
            throw new Error("scene directional light did not sweep with elapsed time");
        const second = await sampleCycle();
        if (
            second.elapsed !== first.elapsed ||
            second.end.some((value, i) => value !== first.end[i])
        )
            throw new Error("rebuild changed the scene light at the same elapsed time");
    },
);
