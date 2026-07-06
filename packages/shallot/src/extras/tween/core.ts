import { getEasing } from "./easing";

// #doc:dev
// ### The timing atom
//
// The tween core is the Web Animations model as pure functions over numbers — no ECS, no entity, no
// stored state. `sample(t)` maps a local time to the value to write; `owns(t)` decides whether to write
// it at all (the `Fill` tails). The `Tween` component and `TweenSystem` drive these, and the
// sequence layer positions `elapsed` — the atom itself just answers "what value, and does it apply".
//
// Build a custom effect kind against it: a spring, a scripted camera move, an animation over something
// that isn't an entity field. Call `sample` / `owns` over your own time source and reuse the shared
// easing surface (`getEasing`, `EASING_FUNCTIONS`). The semantics follow WAAPI (composite + fill +
// timing), so the model is proven rather than bespoke.

export { EASING_FUNCTIONS, type Easing, getEasing, getEasingIndex, getEasingName } from "./easing";

/** how a sampled value combines with the field's current value (WAAPI composite). */
export const Composite = {
    // overwrite the field. the default.
    Replace: 0,
    // add the sampled value to the field's current value as a transient delta.
    // the base stays authoritative, so an `add` tween over a field a sim also
    // writes (a recoil, a flash) stops fighting it — completion with a delta of
    // zero leaves the base untouched. the SSOT relaxation, made explicit.
    Add: 1,
} as const;

/**
 * the window over which the tween owns its field outside the active interval
 * (WAAPI fill). within `[0, duration]` the tween always owns; fill decides the
 * tails.
 */
export const Fill = {
    // own only during the active interval — transient overlays (recoil, flash).
    None: 0,
    // also hold the end value after duration. the default — today's hold-end.
    Forwards: 1,
    // also hold the start value before zero.
    Backwards: 2,
    Both: 3,
} as const;

/**
 * does the tween write its field at this local time? within the active interval
 * it always does; outside, {@link Fill} governs the tails. the clock decides
 * whether a tween runs at all — this decides whether a running tween's value
 * applies now.
 * @example owns(1.5, 1, Fill.Forwards) // true — the after tail holds the end
 */
export function owns(elapsed: number, duration: number, fill: number): boolean {
    if (elapsed < 0) return fill === Fill.Backwards || fill === Fill.Both;
    if (elapsed >= duration) return fill === Fill.Forwards || fill === Fill.Both;
    return true;
}

/**
 * the value to write at this local time: timing → eased progress → from/to lerp
 * → composite. progress saturates at the interval ends, so the before tail
 * samples the start and the after tail (and an instant `duration <= 0`) the end.
 * `base` is the field's current value, read only for {@link Composite.Add}.
 * @example sample(0.5, 1, 0, 0, 100, Composite.Replace, 0) // 50
 */
export function sample(
    elapsed: number,
    duration: number,
    easing: number,
    from: number,
    to: number,
    composite: number,
    base: number,
): number {
    const progress = elapsed >= duration ? 1 : elapsed <= 0 ? 0 : elapsed / duration;
    const eased = getEasing(easing)(progress);
    const value = from + (to - from) * eased;
    return composite === Composite.Add ? base + value : value;
}
