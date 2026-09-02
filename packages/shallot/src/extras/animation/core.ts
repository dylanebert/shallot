import { Composite, Fill, getEasingIndex, owns, sample } from "../tween/core";

export {
    Composite,
    EASING_FUNCTIONS,
    type Easing,
    Fill,
    getEasing,
    getEasingIndex,
    getEasingName,
    owns,
    sample,
} from "../tween/core";

/** a reusable flat channel-to-value animation result. */
export class Pose extends Map<string, number> {}

/** a duration-bearing animation evaluator that is pure over time. */
export interface Playable {
    /** the playable's local duration in seconds. */
    duration: number;
    /** replace `pose` with the channels owned at `t`. */
    evaluate(t: number, pose: Pose): void;
}

/** one numeric key on a normalized clip timeline. */
export interface Keyframe {
    /** normalized position in `[0, 1]`; omitted positions are distributed evenly. */
    offset?: number;
    /** channel value at this key. */
    value: number;
    /** easing from this key to the next. */
    easing?: number | string;
    /** how this segment's sample combines with its zero base. */
    composite?: number | "replace" | "add";
}

/** timing shared by every channel in a keyframe clip. */
export interface KeyframeOptions {
    /** clip duration in seconds. */
    duration: number;
    /** ownership outside the active interval. */
    fill?: number | "none" | "forwards" | "backwards" | "both";
}

const fills = {
    none: Fill.None,
    forwards: Fill.Forwards,
    backwards: Fill.Backwards,
    both: Fill.Both,
} as const;

function fillValue(fill: KeyframeOptions["fill"]): number {
    return typeof fill === "string" ? fills[fill] : (fill ?? Fill.Forwards);
}

function offset(key: Keyframe, index: number, count: number): number {
    return key.offset ?? (count === 1 ? 0 : index / (count - 1));
}

/** build a WAAPI-style numeric keyframe clip over flat channels. */
export function keyframes(
    channels: Readonly<Record<string, readonly Keyframe[]>>,
    options: KeyframeOptions,
): Playable {
    const duration = Math.max(0, options.duration);
    const fill = fillValue(options.fill);
    return {
        duration,
        evaluate(t, pose) {
            pose.clear();
            if (!owns(t, duration, fill)) return;
            const progress = duration <= 0 || t >= duration ? 1 : t <= 0 ? 0 : t / duration;
            for (const [channel, keys] of Object.entries(channels)) {
                if (keys.length === 0) continue;
                let index = keys.length - 1;
                for (let i = 0; i < keys.length - 1; i++) {
                    if (progress < offset(keys[i + 1], i + 1, keys.length)) {
                        index = i;
                        break;
                    }
                }
                if (index === keys.length - 1) {
                    pose.set(channel, keys[index].value);
                    continue;
                }
                const from = keys[index];
                const to = keys[index + 1];
                const start = offset(from, index, keys.length);
                const end = offset(to, index + 1, keys.length);
                const local = end === start ? 1 : (progress - start) / (end - start);
                const easing =
                    typeof from.easing === "string"
                        ? getEasingIndex(from.easing)
                        : (from.easing ?? 0);
                const composite =
                    from.composite === "add" || from.composite === Composite.Add
                        ? Composite.Add
                        : Composite.Replace;
                pose.set(channel, sample(local, 1, easing, from.value, to.value, composite, 0));
            }
        },
    };
}

/** build an arbitrary pure animation clip. */
export function script(fn: (t: number, pose: Pose) => void, duration: number): Playable {
    return {
        duration: Math.max(0, duration),
        evaluate(t, pose) {
            pose.clear();
            fn(t, pose);
        },
    };
}

/** one positioned playable in a mixer track. */
export interface Strip {
    /** child playable. */
    playable: Playable;
    /** parent time at which the strip begins. */
    start: number;
    /** child-local time at the strip's beginning. */
    offset?: number;
    /** parent seconds per child second. */
    scale?: number;
    /** how the strip combines with lower layers. */
    blend?: "replace" | "add";
}

/** an ordered set of strips; later tracks layer over earlier tracks. */
export type Track = readonly Strip[];

/** build a nestable track mixer with positioned and scaled strips. */
export function mixer(tracks: readonly Track[]): Playable {
    let duration = 0;
    for (const track of tracks) {
        for (const strip of track) {
            const scale = strip.scale ?? 1;
            const offset = strip.offset ?? 0;
            duration = Math.max(
                duration,
                strip.start + Math.max(0, strip.playable.duration - offset) * scale,
            );
        }
    }
    const sampled = new Pose();
    return {
        duration,
        evaluate(t, pose) {
            pose.clear();
            for (const track of tracks) {
                for (const strip of track) {
                    const scale = strip.scale ?? 1;
                    if (scale <= 0 || t < strip.start) continue;
                    const local = (strip.offset ?? 0) + (t - strip.start) / scale;
                    if (local > strip.playable.duration) continue;
                    strip.playable.evaluate(local, sampled);
                    for (const [channel, value] of sampled) {
                        pose.set(
                            channel,
                            strip.blend === "add" ? (pose.get(channel) ?? 0) + value : value,
                        );
                    }
                }
            }
        },
    };
}
