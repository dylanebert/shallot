import type { State } from "./state";

const EMPTY: readonly never[] = Object.freeze([]);

const ALL_HANDLES: Array<(state: State) => void> = [];

/**
 * typed handle for a frame-scoped event channel
 * @expand
 */
export interface Events<T> {
    readonly name: string;
    send(state: State, event: T): void;
    read(state: State): readonly T[];
}

/**
 * declare a typed event channel
 * @example
 * const Damage = events<{ amount: number }>("damage");
 * Damage.send(state, { amount: 10 });
 * for (const ev of Damage.read(state)) console.log(ev.amount);
 */
export function events<T>(name: string): Events<T> {
    const queues = new WeakMap<State, T[]>();

    ALL_HANDLES.push((state) => {
        const q = queues.get(state);
        if (q !== undefined) q.length = 0;
    });

    return {
        name,
        send(state, event) {
            let q = queues.get(state);
            if (q === undefined) {
                q = [];
                queues.set(state, q);
            }
            q.push(event);
        },
        read(state) {
            return queues.get(state) ?? EMPTY;
        },
    };
}

/** internal: drain every event channel for a state. invoked at end-of-step and on dispose. */
export function clearAllEvents(state: State): void {
    for (let i = 0; i < ALL_HANDLES.length; i++) ALL_HANDLES[i](state);
}
