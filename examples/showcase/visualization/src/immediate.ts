import type { Plugin, System } from "@dylanebert/shallot";
import { arrow, LinesPlugin, segment } from "@dylanebert/shallot";
import { start } from "./boot";

// The scale path: a swirling flow field of immediate arrows (hundreds, one draw) plus a sweeping sine
// ribbon of segments. Both are rebuilt every frame — immediate geometry clears each frame, so motion is
// free: just emit a different set next frame.
const N = 12;
const SPAN = 9;
const STEP = (SPAN * 2) / (N - 1);

const FeedSystem: System = {
    group: "simulation",
    annotations: { mode: "always" },
    update() {
        const t = performance.now() * 0.001;
        for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
                const x = -SPAN + i * STEP;
                const z = -SPAN + j * STEP;
                const a = t + 0.35 * Math.hypot(x, z) + Math.atan2(z, x);
                arrow([x, 0.05, z], [x + Math.cos(a), 0.05, z + Math.sin(a)], 0x66ccff, 2);
            }
        }
        let px = -SPAN;
        let py = 3 + Math.sin(px * 0.7 + t * 3);
        for (let k = 1; k <= 80; k++) {
            const x = -SPAN + (k / 80) * SPAN * 2;
            const y = 3 + Math.sin(x * 0.7 + t * 3);
            segment([px, py, 0], [x, y, 0], 0xffcc44, 3);
            px = x;
            py = y;
        }
    },
};

const FeedPlugin: Plugin = {
    name: "ImmediateFeed",
    systems: [FeedSystem],
    dependencies: [LinesPlugin],
};

await start([FeedPlugin], "../scenes/immediate.scene");
