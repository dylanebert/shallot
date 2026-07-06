import {
    entity,
    f32,
    type Plugin,
    type Single,
    type State,
    type System,
    sparse,
    u8,
    u32,
} from "../../engine";
import { camel, getComponent, kebab, lanes } from "../../engine/ecs/core";
import { Composite, Fill, getEasingIndex, getEasingName, owns, sample } from "./core";

/** play control for a {@link Tween} or {@link Sequence}. set `state` to drive it; the system advances `elapsed`. */
export const TweenState = {
    // not advancing — paused, or not started. the system writes nothing.
    Idle: 0,
    // advancing `elapsed` each frame, writing the field.
    Playing: 1,
    // reached the end. holds per fill; set back to Idle + reset `elapsed` to replay.
    Complete: 2,
} as const;

/**
 * animate one numeric field of a target over time, with an easing curve
 *
 * @example
 * ```
 * <a tween="target: @ball; field: transform.pos.y; to: 5; duration: 0.4; easing: ease-out-quad" />
 * ```
 */
export const Tween = {
    /** play control; a fresh tween defaults to playing */
    state: sparse(u8),
    /** the field path this tween writes, `component.field[.lane]` (`transform.pos.y`) */
    field: sparse(u32),
    /** start value; NaN captures the field's current value on the first frame (the `.to` convenience) */
    from: sparse(f32),
    /** end value at `duration` */
    to: sparse(f32),
    /** length of the animation, seconds */
    duration: sparse(f32),
    /** current clock position, seconds; seekable — set it to scrub */
    elapsed: sparse(f32),
    /** easing curve, by id (the `ease-out-quad` names) */
    easing: sparse(u8),
    /** how the sampled value applies to the field: `replace` overwrites it, `add` layers a transient delta */
    composite: sparse(u8),
    /** whether the tween holds its value outside the active interval — `none`, `forwards`, `backwards`, `both` */
    fill: sparse(u8),
    /** rewind and replay on completion */
    loop: sparse(u8),
    /** the timeline this tween plays on, or 0 for standalone; resolves via `@name` in a scene */
    sequence: sparse(entity),
    /** start offset on the sequence's playhead, seconds */
    at: sparse(f32),
    /** the entity whose field is animated; resolves via `@name` in a scene */
    target: sparse(entity),
};

/**
 * a timeline that plays {@link Tween}s placed on it with `at` — a shared playhead many tweens read
 */
export const Sequence = {
    /** play control; a fresh sequence defaults to playing */
    state: sparse(u8),
    /** playhead position, seconds; the tweens on it read their slice as `elapsed - at` */
    elapsed: sparse(f32),
    /** timeline period, seconds; grows to span its tweens' ends */
    duration: sparse(f32),
    /** wrap the whole timeline back to the start on completion */
    loop: sparse(u8),
};

// "component.field[.lane]" path interning: a stable id per path, the typed
// Single it writes resolved once at intern time, and the path back for format.
// `tween()` and scene parse both bind through `internField`, so the programmatic
// and declarative surfaces share one table — the seam where a path becomes a
// setter (parse-time bind, not runtime reflection). `Tween.field` stores the id;
// the hot path is one array index + one virtual `.set`. reset in `initialize`,
// since the setters resolve against the registered components (rebuilt per app).
//
// id 0 is permanently the empty path — the `Tween.field` default + "no field"
// sentinel — seeded here at module load, not only in `initialize`, so the invariant
// holds in every process that imports the module. The scene formatter
// (`scripts/format.ts`) registers components but never runs plugin `initialize`;
// without the module-scope seed, the first real path it interns would take id 0 and
// collide with the default, so its `stripDefaults` pass would wrongly elide an
// authored `field` (it once stripped `field: transform.pos.y` off the tween showcase).
const fieldIds = new Map<string, number>([["", 0]]);
const fieldPaths: string[] = [""];
const fieldSetters: (Single | null)[] = [null];

function internField(path: string): number {
    let id = fieldIds.get(path);
    if (id !== undefined) return id;
    id = fieldPaths.length;
    fieldIds.set(path, id);
    fieldPaths.push(path);
    fieldSetters.push(resolveSetter(path));
    return id;
}

// resolve "component.field" (a scalar Single) or "component.field.lane" (a lane
// of a Pair/Quad) to the typed Single it writes. every kitchen field is typed,
// so there's no number[] / proxy ladder — just the one accessor, or null when
// the path doesn't name a scalar field. component + field names normalize
// (kebab in the registry, camel for the field), the lane is literal x/y/z/w.
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

// wrap a looping playhead back into [0, period); modulo so a long frame can't
// overshoot past one cycle. period <= 0 is degenerate (instant) — restart at 0.
const wrap = (elapsed: number, period: number): number => (period > 0 ? elapsed % period : 0);

const TweenSystem: System = {
    name: "tween",
    group: "simulation",
    // grow each sequence's period to span its children. `tween()` does this
    // eagerly as it attaches a child; a scene-authored tween only sets the
    // fields, so the first frame after load reconciles every sequence here (the
    // GSAP model: the period is the last child's end). idempotent with the
    // eager grow, so a mixed scene + code timeline lands the same period.
    setup(state) {
        for (const eid of state.query([Tween])) {
            const seq = Tween.sequence.get(eid);
            if (seq === 0) continue;
            const end = Tween.at.get(eid) + Tween.duration.get(eid);
            if (end > Sequence.duration.get(seq)) Sequence.duration.set(seq, end);
        }
    },
    update(state) {
        const dt = state.time.deltaTime;

        // advance the sequence clocks first, so sequenced tweens read this frame's playhead
        for (const seq of state.query([Sequence])) {
            if (Sequence.state.get(seq) !== TweenState.Playing) continue;
            const period = Sequence.duration.get(seq);
            let elapsed = Sequence.elapsed.get(seq) + dt;
            if (elapsed >= period) {
                if (Sequence.loop.get(seq)) {
                    elapsed = wrap(elapsed, period);
                } else {
                    elapsed = period;
                    Sequence.state.set(seq, TweenState.Complete);
                }
            }
            Sequence.elapsed.set(seq, elapsed);
        }

        for (const eid of state.query([Tween])) {
            const duration = Tween.duration.get(eid);
            const seq = Tween.sequence.get(eid);
            let elapsed: number;

            if (seq !== 0) {
                // the clock owns time; this tween reads its slice of the playhead
                if (Sequence.state.get(seq) === TweenState.Idle) continue;
                elapsed = Sequence.elapsed.get(seq) - Tween.at.get(eid);
            } else {
                const tweenState = Tween.state.get(eid);
                if (tweenState === TweenState.Idle) continue;
                elapsed = Tween.elapsed.get(eid);
                if (tweenState === TweenState.Playing) {
                    elapsed += dt;
                    if (elapsed >= duration) {
                        if (Tween.loop.get(eid)) {
                            elapsed = wrap(elapsed, duration);
                        } else {
                            elapsed = duration;
                            Tween.state.set(eid, TweenState.Complete);
                        }
                    }
                    Tween.elapsed.set(eid, elapsed);
                }
            }

            if (!owns(elapsed, duration, Tween.fill.get(eid))) continue;

            const setter = fieldSetters[Tween.field.get(eid)];
            if (!setter) continue;
            const target = Tween.target.get(eid);
            const composite = Tween.composite.get(eid);

            // capture the start once, on the first owning frame: the current
            // field value for `replace` (the `.to` convenience), zero for `add`
            // (the delta starts at none). stored in `from`, so a seek replays it.
            let from = Tween.from.get(eid);
            if (Number.isNaN(from)) {
                from = composite === Composite.Add ? 0 : setter.get(target);
                Tween.from.set(eid, from);
            }

            const base = composite === Composite.Add ? setter.get(target) : 0;
            setter.set(
                target,
                sample(
                    elapsed,
                    duration,
                    Tween.easing.get(eid),
                    from,
                    Tween.to.get(eid),
                    composite,
                    base,
                ),
            );
        }
    },
};

// string → enum value, derived from the developer enums so the helper and the
// scene `enums` traits share one source of truth (both kebab the enum keys).
const named = (e: Record<string, number>): Record<string, number> => {
    const m: Record<string, number> = {};
    for (const k in e) m[kebab(k)] = e[k];
    return m;
};
const composites = named(Composite);
const fills = named(Fill);

/** options for {@link tween}. `to` is required; the rest default to a 1s linear replace-with-hold-end. */
export interface TweenOptions {
    readonly to: number;
    readonly from?: number;
    readonly duration?: number;
    readonly easing?: string;
    readonly composite?: "replace" | "add";
    readonly fill?: "none" | "forwards" | "backwards" | "both";
    readonly loop?: boolean;
    readonly sequence?: number;
    readonly at?: number;
}

/**
 * animate one numeric field of `target`, returning the tween entity (null if `path` names no scalar field)
 * @example tween(state, ball, "transform.pos.y", { to: 5, duration: 0.4, easing: "ease-out-quad" })
 */
export function tween(
    state: State,
    target: number,
    path: string,
    opts: TweenOptions,
): number | null {
    const field = internField(path);
    if (!fieldSetters[field]) return null;

    const eid = state.create();
    state.add(eid, Tween);

    const duration = opts.duration ?? 1;
    const at = opts.at ?? 0;
    const seq = opts.sequence ?? 0;

    Tween.field.set(eid, field);
    Tween.target.set(eid, target);
    Tween.to.set(eid, opts.to);
    Tween.from.set(eid, opts.from ?? Number.NaN);
    Tween.duration.set(eid, duration);
    Tween.easing.set(eid, getEasingIndex(opts.easing ?? "linear"));
    Tween.composite.set(eid, composites[opts.composite ?? "replace"]);
    Tween.fill.set(eid, fills[opts.fill ?? "forwards"]);
    Tween.loop.set(eid, opts.loop ? 1 : 0);
    Tween.sequence.set(eid, seq);
    Tween.at.set(eid, at);

    // state defaults to Playing; a sequenced tween reads its clock, not its own
    // state, so the default just stands. grow the timeline to span this child
    // (GSAP: the period is the last child's end).
    if (seq !== 0 && at + duration > Sequence.duration.get(seq)) {
        Sequence.duration.set(seq, at + duration);
    }
    return eid;
}

/**
 * create a {@link Sequence} timeline and return its entity; place tweens on it with `tween(…, { sequence, at })`
 * @example const s = sequence(state, { loop: true }); tween(state, e, "transform.pos.y", { to: 2, at: 0, duration: 1, fill: "none", sequence: s })
 */
export function sequence(state: State, opts: { loop?: boolean } = {}): number {
    const eid = state.create();
    state.add(eid, Sequence);
    Sequence.loop.set(eid, opts.loop ? 1 : 0);
    return eid;
}

/** field animation via the {@link Tween} + {@link Sequence} components; add it to author or run tweens */
export const TweenPlugin: Plugin = {
    name: "Tween",
    components: { Tween, Sequence },
    systems: [TweenSystem],
    traits: {
        Tween: {
            defaults: () => ({
                state: TweenState.Playing,
                field: 0,
                from: Number.NaN,
                to: 0,
                duration: 1,
                elapsed: 0,
                easing: 0,
                composite: Composite.Replace,
                fill: Fill.Forwards,
                loop: 0,
                sequence: 0,
                at: 0,
                target: 0,
            }),
            parse: { easing: getEasingIndex, field: internField },
            format: { easing: getEasingName, field: (id: number) => fieldPaths[id] ?? "" },
            enums: { state: TweenState, composite: Composite, fill: Fill },
        },
        Sequence: {
            defaults: () => ({ state: TweenState.Playing, elapsed: 0, duration: 0, loop: 0 }),
            enums: { state: TweenState },
        },
    },
    initialize() {
        // reset per app build so setters re-resolve against the rebuilt components,
        // preserving the id-0 empty-path sentinel (the Tween.field default, no setter)
        fieldIds.clear();
        fieldIds.set("", 0);
        fieldPaths.length = 0;
        fieldPaths.push("");
        fieldSetters.length = 0;
        fieldSetters.push(null);
    },
};
