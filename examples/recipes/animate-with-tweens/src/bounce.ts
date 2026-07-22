import { type Plugin, type State, sequence, tween } from "@dylanebert/shallot";

// `tween()` animates a field and returns the tween entity. Passing `to` alone captures the field's
// current value as the start (the GSAP `.to` convenience), so the coin rises from wherever it sits.
// `sequence()` builds a looping timeline the same way the scene's `<a sequence>` does, with an
// `ease-out-back` rise that overshoots the top before settling.
export const Bounce = {};

export const Bouncer = {
    name: "Bouncer",
    components: { Bounce },
    traits: { Bounce: { defaults: () => ({}) } },
    warm(state: State) {
        for (const coin of state.query([Bounce])) {
            const timeline = sequence(state, { loop: true });
            tween(state, coin, "transform.pos.y", {
                to: 2.4,
                duration: 0.7,
                easing: "ease-out-back",
                fill: "none",
                sequence: timeline,
            });
            tween(state, coin, "transform.pos.y", {
                from: 2.4,
                to: 0.6,
                at: 0.7,
                duration: 0.7,
                easing: "ease-in-quad",
                fill: "none",
                sequence: timeline,
            });
        }
    },
} satisfies Plugin;

export default Bouncer;
