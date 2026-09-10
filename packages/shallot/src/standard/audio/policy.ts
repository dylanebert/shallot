// SFX policy — a per-name instance-limit contract over play(), FMOD/Wwise style.
// The kernel cull is the audibility half (it quiets the least-audible voice when
// the 64-voice pool is full); this is the per-event budget half it lacks: a
// max-instances cap, a min-interval cooldown, and a steal mode. Pure
// registration + a pure cooldown predicate; SoundSystem's play() enforces it.

/** a per-name SFX limit, the argument to {@link sfx}. All fields optional */
export interface SfxPolicy {
    /** max simultaneous instances of this sound; 0 / omitted = unlimited */
    max?: number;
    /** min seconds between triggers; a re-trigger inside the window is dropped */
    cooldown?: number;
    /** at the cap, which playing instance to cull: the oldest, the quietest, or none (drop the new trigger). default "oldest" */
    steal?: "oldest" | "quietest" | "none";
}

const policies = new Map<string, Required<SfxPolicy>>();
// per-name elapsed-time of the last admitted trigger. The cooldown clock is
// state.time.elapsed, so a State rebuild resets it to 0 — withinCooldown reads a
// backwards clock (elapsed < last) as expired, self-healing without a hook
const fired = new Map<string, number>();

/**
 * declare a per-name instance limit for {@link play}, FMOD/Wwise style: cap the
 * simultaneous instances, enforce a min gap between triggers, and pick the cull
 * victim at the cap. `play(name)` consults it automatically; a name with no
 * policy plays unbounded as before. Re-registering a name overwrites its policy
 * @example
 * sfx("coin", { max: 8, cooldown: 0.05, steal: "oldest" });
 * play(state, "coin"); // capped, cooled, and stolen-from per the policy
 */
export function sfx(name: string, policy: SfxPolicy): void {
    policies.set(name, {
        max: policy.max ?? 0,
        cooldown: policy.cooldown ?? 0,
        steal: policy.steal ?? "oldest",
    });
}

/** the resolved policy for a name, or undefined when none is registered */
export function policyFor(name: string): Required<SfxPolicy> | undefined {
    return policies.get(name);
}

/** true when `name` last fired inside its cooldown window; the trigger should drop. A backwards clock (a State rebuild reset elapsed) reads as expired */
export function withinCooldown(name: string, cooldown: number, elapsed: number): boolean {
    if (cooldown <= 0) return false;
    const last = fired.get(name);
    return last !== undefined && elapsed >= last && elapsed - last < cooldown;
}

/** record an admitted trigger's time, opening the cooldown window */
export function markCooldown(name: string, elapsed: number): void {
    fired.set(name, elapsed);
}
