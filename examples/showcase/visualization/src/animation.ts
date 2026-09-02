import { AnimationPlugin, keyframes, Playables, type Plugin } from "@dylanebert/shallot";
import { start } from "./boot";

const curves = ["linear", "ease-out-cubic", "ease-out-back", "ease-out-elastic", "ease-out-bounce"];
const Clips = {
    name: "CurveClips",
    dependencies: [AnimationPlugin],
    initialize() {
        for (const [i, easing] of curves.entries()) {
            Playables.register({
                name: `curve-${i}`,
                ...keyframes(
                    {
                        "transform.pos.y": [
                            { value: -1.5, offset: 0, easing },
                            { value: 1.5, offset: 0.5, easing },
                            { value: -1.5, offset: 1 },
                        ],
                    },
                    { duration: 2.8 },
                ),
            });
        }
    },
} satisfies Plugin;

await start([Clips], "../scenes/animation.scene");
