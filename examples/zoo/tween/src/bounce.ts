import { type Plugin, type State, sequence, tween } from "@dylanebert/shallot";

// #doc:intro
// Field animation with easing: a tween is a pure f(t) over one numeric field of an entity, following
// the Web Animations model (like GSAP). Author tweens as scene entities in the editor, or create them
// in code with `tween()`.

// #doc:code source:tween/public/scenes/tween.scene
// A `<a tween>` animates one `field` of its `target` from `from` to `to` over `duration`, along an
// `easing` curve. Put two on a looping `<a sequence>` — one rising, one falling at `at: 0.7` — and the
// ball bobs forever. `fill: none` makes each phase own only its own window, so the rise hands off to
// the fall cleanly instead of both fighting over the field.

// #doc:code
// ### From code
//
// `tween()` animates a field and returns the tween entity. Passing `to` alone captures the field's
// current value as the start (the GSAP `.to` convenience), so the coin rises from wherever it sits.
// `sequence()` builds the looping timeline the same way the scene does — an `ease-out-back` rise
// overshoots the top before settling. `Bounce` is a no-field marker naming the entity this plugin drives.
// #region bounce
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
// #endregion

export default Bouncer;
