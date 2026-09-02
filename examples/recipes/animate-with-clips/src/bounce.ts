import {
    AnimationPlugin,
    keyframes,
    mixer,
    Playables,
    type Plugin,
    script,
} from "@dylanebert/shallot";

const rise = keyframes(
    { "transform.pos.y": [{ value: 0.6, easing: "ease-out-back" }, { value: 2.4 }] },
    { duration: 0.7, fill: "none" },
);
const fall = keyframes(
    { "transform.pos.y": [{ value: 2.4, easing: "ease-in-quad" }, { value: 0.6 }] },
    { duration: 0.7, fill: "none" },
);

export const Bounce = {};

export const Bouncer = {
    name: "Bouncer",
    components: { Bounce },
    dependencies: [AnimationPlugin],
    traits: { Bounce: { defaults: () => ({}) } },
    initialize() {
        Playables.register({
            name: "lift",
            ...mixer([
                [
                    { playable: rise, start: 0 },
                    { playable: fall, start: 0.7 },
                ],
            ]),
        });
        Playables.register({
            name: "bounce",
            ...script(
                (t, pose) => pose.set("transform.pos.y", 0.6 + Math.sin(t * Math.PI) * 1.8),
                1,
            ),
        });
    },
} satisfies Plugin;

export default Bouncer;
