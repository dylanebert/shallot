import { f32, Line, LinesPlugin, type Plugin, type System, sparse } from "@dylanebert/shallot";
import { start } from "./boot";

// The declarative side: retained Line/Arrow entities authored in the scene, animated by a custom
// system writing their fields. `Spin` marks a hand and carries its rate + z-wobble; the system rotates
// each marked Line's offset on a fixed-radius circle each frame. Persistent entities mutated through
// their component — the opposite of the immediate API, and the genuinely bespoke part of this demo.
const RADIUS = 2.6;

const Spin = {
    speed: sparse(f32),
    wobble: sparse(f32),
};

const SpinSystem: System = {
    group: "simulation",
    annotations: { mode: "always" },
    update(state) {
        const t = performance.now() * 0.001;
        for (const eid of state.query([Spin, Line])) {
            const a = t * Spin.speed.get(eid);
            const w = Spin.wobble.get(eid);
            Line.offset.set(
                eid,
                Math.cos(a) * RADIUS,
                Math.sin(a) * RADIUS,
                Math.sin(a * 0.7) * RADIUS * w,
                0,
            );
        }
    },
};

const SpinPlugin: Plugin = {
    name: "RetainedSpin",
    components: { Spin },
    systems: [SpinSystem],
    traits: {
        Spin: {
            requires: [Line],
            defaults: () => ({ speed: 1, wobble: 0 }),
        },
    },
    dependencies: [LinesPlugin],
};

await start([SpinPlugin], "../scenes/retained.scene");
