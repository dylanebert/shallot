import { resolve } from "node:path";
import {
    AudioPlugin,
    build,
    composeTransform,
    Listener,
    Sound,
    Time,
    Transform,
} from "@dylanebert/shallot";
import { OrbitPlugin } from "@dylanebert/shallot/extras";
import { check } from "@dylanebert/shallot/harness/check";
import AudioDemo from "./demo";

const SCENE = resolve(import.meta.dir, "../public/scenes/play-sound.scene");

check(
    "play-sound loop follows its orbit across the listener",
    {
        claim: "play-sound's actual looping scene source follows its orbit and crosses the listener-relative side plane",
    },
    async () => {
        const app = await build({
            defaults: false,
            plugins: [AudioPlugin, OrbitPlugin, AudioDemo],
            scene: SCENE,
        });
        try {
            const { state } = app;
            const source = [...state.query([Sound, Transform])].find(
                (eid) => Sound.loop.get(eid) === 1,
            );
            const listener = state.only([Listener, Transform]);
            if (source === undefined)
                throw new Error("actual scene has no looping positional source");
            if (listener < 0) throw new Error("actual scene has no positional listener");

            const world = new Float32Array(16);
            const sample = () => {
                composeTransform(listener, world);
                const dx = Transform.pos.x.get(source) - world[12];
                const dy = Transform.pos.y.get(source) - world[13];
                const dz = Transform.pos.z.get(source) - world[14];
                return {
                    x: Transform.pos.x.get(source),
                    z: Transform.pos.z.get(source),
                    radius: Math.hypot(Transform.pos.x.get(source), Transform.pos.z.get(source)),
                    side: dx * world[0] + dy * world[1] + dz * world[2],
                };
            };

            state.step(0);
            const start = sample();
            for (let i = 0; i < 270; i++) state.step(Time.FIXED_DT);
            const opposite = sample();
            if (Sound.loop.get(source) !== 1) throw new Error("source is not authored as a loop");
            if (Math.abs(start.radius - 4) > 1e-5 || Math.abs(opposite.radius - 4) > 1e-5)
                throw new Error("looping source did not stay on its authored orbit");
            if (Math.hypot(start.x - opposite.x, start.z - opposite.z) < 7.9)
                throw new Error("looping source did not travel around its orbit");
            if (start.side * opposite.side >= 0)
                throw new Error("source did not cross the listener-relative side plane");
        } finally {
            app.dispose();
        }
    },
);
