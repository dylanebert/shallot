import type { Plugin, State } from "../engine";
import { Time } from "../engine/ecs";

/** wire a plugin's systems into state without going through `app({ plugins })` */
export function attach(state: State, plugin: Plugin): void {
    for (const s of plugin.systems ?? []) state.addSystem(s, plugin.name);
}

/** step in clamp-sized chunks so durations past `step()`'s ~67ms dt clamp advance fully */
export function stepFor(state: State, duration: number): void {
    const maxDt = Time.FIXED_DT * Time.MAX_FIXED_STEPS;
    while (duration > maxDt) {
        state.step(maxDt);
        duration -= maxDt;
    }
    if (duration > 0) state.step(duration);
}
