import { pair, traits, ChildOf, type State } from "../../engine";
import { Tween, TweenState, ensureResolved, captureFromValue } from "./tween";

const compareNumbers = (a: number, b: number) => a - b;
const compareByEndTime = (a: { endTime: number }, b: { endTime: number }) => a.endTime - b.endTime;

const childrenBuffer: number[] = [];
const tweensBuffer: { eid: number; endTime: number }[] = [];

const delays = new Map<number, number>();

export function clearDelays(): void {
    delays.clear();
}

export const Pause = {
    duration: [] as number[],
};

traits(Pause, {
    defaults: () => ({ duration: 0.5 }),
});

export const SequenceState = {
    Idle: 0,
    Playing: 1,
    Complete: 2,
} as const;

export const Sequence = {
    state: [] as number[],
    elapsed: [] as number[],
    duration: [] as number[],
};

traits(Sequence, {
    defaults: () => ({
        state: SequenceState.Idle,
        elapsed: 0,
        duration: 0,
    }),
    enums: { state: SequenceState },
});

function sortedChildren(state: State, parentEid: number): number[] {
    childrenBuffer.length = 0;
    for (const childEid of state.query([pair(ChildOf.relation, parentEid)])) {
        childrenBuffer.push(childEid);
    }
    childrenBuffer.sort(compareNumbers);
    return childrenBuffer;
}

export function computeTweenDelays(state: State, seqEid: number): void {
    const children = sortedChildren(state, seqEid);
    let cumulativeDelay = 0;
    let lastTweenEnd = 0;

    for (const childEid of children) {
        if (state.hasComponent(childEid, Pause)) {
            cumulativeDelay += Pause.duration[childEid] ?? 0;
        } else if (state.hasComponent(childEid, Tween)) {
            delays.set(childEid, cumulativeDelay);
            const end = cumulativeDelay + (Tween.duration[childEid] ?? 0);
            if (end > lastTweenEnd) lastTweenEnd = end;
        }
    }

    Sequence.duration[seqEid] = Math.max(lastTweenEnd, cumulativeDelay);
}

export function updateSequences(state: State, dt: number): void {
    for (const eid of state.query([Sequence, AutoPlay])) {
        const s = Sequence.state[eid];
        if (s === SequenceState.Idle) {
            Sequence.state[eid] = SequenceState.Playing;
        } else if (s === SequenceState.Complete && AutoPlay.loop[eid]) {
            resetSequence(state, eid);
            Sequence.state[eid] = SequenceState.Playing;
        }
    }

    for (const seqEid of state.query([Sequence])) {
        if (Sequence.state[seqEid] !== SequenceState.Playing) continue;

        const prevElapsed = Sequence.elapsed[seqEid] ?? 0;

        if (prevElapsed === 0) {
            computeTweenDelays(state, seqEid);
        }
        const elapsed = prevElapsed + dt;
        Sequence.elapsed[seqEid] = elapsed;

        for (const childEid of state.query([pair(ChildOf.relation, seqEid), Tween])) {
            if (Tween.state[childEid] !== TweenState.Idle) continue;

            const delay = delays.get(childEid) ?? 0;
            const shouldStart = elapsed >= delay;
            const wasStarted = prevElapsed >= delay;

            if (shouldStart) {
                captureFromValue(state, childEid);
                Tween.state[childEid] = TweenState.Playing;
                Tween.elapsed[childEid] = wasStarted ? 0 : elapsed - delay - dt;
            }
        }
    }
}

function resolve(state: State, seqEid: number): void {
    computeTweenDelays(state, seqEid);
    tweensBuffer.length = 0;

    for (const childEid of state.query([pair(ChildOf.relation, seqEid), Tween])) {
        if (
            Tween.state[childEid] === TweenState.Complete &&
            Tween.elapsed[childEid] >= Tween.duration[childEid]
        ) {
            continue;
        }
        const delay = delays.get(childEid) ?? 0;
        const duration = Tween.duration[childEid] ?? 0;
        tweensBuffer.push({ eid: childEid, endTime: delay + duration });
    }

    tweensBuffer.sort(compareByEndTime);

    for (const { eid } of tweensBuffer) {
        Tween.state[eid] = TweenState.Complete;
        ensureResolved(state, eid);
    }
}

export function resolveSequence(state: State, seqEid: number): void {
    if (Sequence.state[seqEid] === SequenceState.Complete) return;
    Sequence.state[seqEid] = SequenceState.Complete;
    resolve(state, seqEid);
}

export function resetSequence(state: State, seqEid: number): void {
    Sequence.state[seqEid] = SequenceState.Idle;
    Sequence.elapsed[seqEid] = 0;
    Sequence.duration[seqEid] = 0;
    for (const childEid of state.query([pair(ChildOf.relation, seqEid), Tween])) {
        Tween.state[childEid] = TweenState.Idle;
        Tween.elapsed[childEid] = 0;
    }
}

export function resolveAll(state: State): void {
    for (const seqEid of state.query([Sequence])) {
        if (Sequence.state[seqEid] === SequenceState.Complete) {
            resolve(state, seqEid);
        }
    }
}

export const AutoPlay = {
    loop: [] as number[],
};
traits(AutoPlay, {
    defaults: () => ({ loop: 0 }),
});

export function checkCompletion(state: State): void {
    for (const seqEid of state.query([Sequence])) {
        if (Sequence.state[seqEid] !== SequenceState.Playing) continue;

        let allComplete = true;
        let hasChildren = false;

        for (const childEid of state.query([pair(ChildOf.relation, seqEid), Tween])) {
            hasChildren = true;
            if (Tween.state[childEid] !== TweenState.Complete) {
                allComplete = false;
                break;
            }
        }

        const elapsed = Sequence.elapsed[seqEid] ?? 0;
        const duration = Sequence.duration[seqEid] ?? 0;

        if (hasChildren && allComplete && elapsed >= duration) {
            Sequence.state[seqEid] = SequenceState.Complete;
        }
    }
}
