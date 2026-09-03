import { entity, f32, type Plugin, type Single, type System, sparse, u8, u32 } from "../../engine";
import { camel, getComponent, lanes } from "../../engine/ecs/core";
import { Registry } from "../../engine/utils";
import { type Playable, Pose } from "./core";

/** play control for an {@link Animator}. */
export const AnimationState = {
    /** hold the current time while continuing to evaluate it for seeking. */
    Idle: 0,
    /** advance and evaluate each simulation step. */
    Playing: 1,
    /** hold the playable's final evaluation. */
    Complete: 2,
} as const;

/** a named playable registered for animator lookup. */
export interface PlayableEntry extends Playable {
    /** scene-facing clip name. */
    name: string;
}

/** every animation playable, keyed by its scene-facing name. */
export const Playables = new Registry<PlayableEntry>();

/** scene-facing animation binding and playhead. */
export const Animator = {
    /** registered playable name, interned to a numeric id. */
    clip: sparse(u32),
    /** entity receiving the pose, or the animator entity when omitted. */
    target: sparse(entity),
    /** current playhead position in seconds; set directly to seek. */
    time: sparse(f32),
    /** play control. */
    state: sparse(u8),
    /** playhead speed multiplier. */
    scale: sparse(f32),
    /** rewind and replay at the playable's duration. */
    loop: sparse(u8),
};

function clipId(name: string): number {
    const id = Playables.id(name);
    return id === undefined ? 0 : id + 1;
}

function clipName(id: number): string {
    return id === 0 ? "" : (Playables.name(id - 1) ?? "");
}

function resolveSetter(path: string): Single | null {
    const parts = path.split(".");
    if (parts.length < 2 || parts.length > 3) return null;
    const component = getComponent(parts[0]);
    if (!component) return null;
    const field = (component as Record<string, unknown>)[camel(parts[1])];
    if (parts.length === 2) return lanes(field) === 1 ? (field as Single) : null;
    if (lanes(field) < 2) return null;
    const lane = (field as Record<string, unknown>)[parts[2]];
    return lanes(lane) === 1 ? (lane as Single) : null;
}

const pose = new Pose();
const setters = new Map<string, Single | null>();

/** advances animator playheads, evaluates their playables, and writes resolved component lanes. */
export const AnimationSystem: System = {
    name: "animation",
    group: "simulation",
    update(state) {
        for (const eid of state.query([Animator])) {
            const entry = Playables.get(clipName(Animator.clip.get(eid)));
            if (!entry) continue;

            let time = Animator.time.get(eid);
            if (Animator.state.get(eid) === AnimationState.Playing) {
                time += state.time.deltaTime * Animator.scale.get(eid);
                if (time >= entry.duration) {
                    if (Animator.loop.get(eid)) {
                        time = entry.duration > 0 ? time % entry.duration : 0;
                    } else {
                        time = entry.duration;
                        Animator.state.set(eid, AnimationState.Complete);
                    }
                }
                Animator.time.set(eid, time);
            }
            entry.evaluate(time, pose);
            const target = Animator.target.get(eid) || eid;
            for (const [path, value] of pose) {
                let setter = setters.get(path);
                if (setter === undefined) {
                    setter = resolveSetter(path);
                    setters.set(path, setter);
                }
                setter?.set(target, value);
            }
        }
    },
};

/** animation registry, animator component, and simulation binding system. */
export const AnimationPlugin: Plugin = {
    name: "Animation",
    components: { Animator },
    systems: [AnimationSystem],
    traits: {
        Animator: {
            defaults: () => ({
                clip: 0,
                target: 0,
                time: 0,
                state: AnimationState.Playing,
                scale: 1,
                loop: 0,
            }),
            parse: { clip: clipId },
            format: { clip: clipName },
            enums: { state: AnimationState },
        },
    },
    initialize() {
        Playables.clear();
        setters.clear();
    },
};

export {
    Composite,
    type Easing,
    Fill,
    type Keyframe,
    type KeyframeOptions,
    keyframes,
    mixer,
    type Playable,
    Pose,
    type Strip,
    script,
    type Track,
} from "./core";
