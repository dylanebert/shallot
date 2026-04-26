import { traits, type State, type System, type Plugin } from "../../engine";
import { Sequence, SequenceState, TweenSystem, resolveSequence, resetSequence } from "../tween";

export const Timeline = {
    step: [] as number[],
    target: [] as number[],
};

traits(Timeline, { defaults: () => ({ step: 0, target: 0 }) });

export const Transition = {
    from: [] as number[],
    to: [] as number[],
};

traits(Transition, { defaults: () => ({ from: 0, to: 0 }) });

const TimelineSystem: System = {
    group: "simulation",
    before: [TweenSystem],
    update(state: State) {
        for (const eid of state.query([Timeline])) {
            const current = Timeline.step[eid];
            const target = Timeline.target[eid];
            if (current === target) continue;

            for (const seqEid of state.query([Sequence, Transition])) {
                if (Sequence.state[seqEid] === SequenceState.Playing) {
                    resolveSequence(state, seqEid);
                }
            }

            for (const seqEid of state.query([Sequence, Transition])) {
                if (Transition.from[seqEid] === current && Transition.to[seqEid] === target) {
                    resetSequence(state, seqEid);
                    Sequence.state[seqEid] = SequenceState.Playing;
                    break;
                }
            }

            Timeline.step[eid] = target;
        }
    },
};

export const TimelinePlugin: Plugin = {
    name: "Timeline",
    components: { Timeline, Transition },
    systems: [TimelineSystem],
};
