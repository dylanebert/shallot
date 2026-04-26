import { Target, traits, type Component, type State, type System, type Plugin } from "../../engine";
import { toCamelCase, getComponent } from "../../engine/ecs/core";

interface FieldAccessor {
    get(eid: number): number;
    set(eid: number, value: number): void;
}
import { getEasingIndex, getEasing } from "./easing";
import {
    Sequence,
    Pause,
    AutoPlay,
    updateSequences,
    resolveAll,
    checkCompletion,
    clearDelays,
} from "./sequence";

interface AccessorEntry {
    accessor: FieldAccessor;
    path: string;
}

const accessors = new Map<number, AccessorEntry>();
const fromValues = new Map<number, number>();

function resolveFieldPath(path: string): { component: string; field: string } | null {
    const dotIndex = path.lastIndexOf(".");
    if (dotIndex === -1) return null;
    return {
        component: path.slice(0, dotIndex),
        field: path.slice(dotIndex + 1),
    };
}

function bindFieldAccessor(
    bindingId: number,
    componentName: string,
    fieldPath: string,
): FieldAccessor | null {
    const registered = getComponent(componentName);
    if (!registered) return null;

    const camelPath = toCamelCase(fieldPath);
    const field = (registered.component as Component)[camelPath];
    if (field == null) return null;

    let accessor: FieldAccessor;

    if (
        typeof field === "object" &&
        typeof (field as FieldAccessor).get === "function" &&
        typeof (field as FieldAccessor).set === "function"
    ) {
        accessor = field as FieldAccessor;
    } else if (ArrayBuffer.isView(field) || Array.isArray(field)) {
        const data = field as number[];
        accessor = {
            get: (eid) => data[eid],
            set: (eid, v) => {
                data[eid] = v;
            },
        };
    } else {
        return null;
    }

    const path = `${componentName}.${fieldPath}`;
    accessors.set(bindingId, { accessor, path });
    return accessor;
}

function getOrBindAccessor(tweenEid: number): FieldAccessor | undefined {
    const path = Tween.field[tweenEid];
    if (!path) return undefined;

    const existing = accessors.get(tweenEid);
    if (existing && existing.path === path) return existing.accessor;

    const parsed = resolveFieldPath(path);
    if (!parsed) return undefined;

    return bindFieldAccessor(tweenEid, parsed.component, parsed.field) ?? undefined;
}

export const TweenState = {
    Idle: 0,
    Playing: 1,
    Complete: 2,
} as const;

const fieldPaths = new Map<number, string>();

function fieldProxy(): Record<number, string | undefined> {
    return new Proxy({} as Record<number, string | undefined>, {
        get(_, prop) {
            const eid = Number(prop);
            if (Number.isNaN(eid)) return undefined;
            return fieldPaths.get(eid);
        },
        set(_, prop, value) {
            const eid = Number(prop);
            if (Number.isNaN(eid)) return false;
            if (value === undefined || value === null) {
                fieldPaths.delete(eid);
            } else {
                fieldPaths.set(eid, value as string);
            }
            return true;
        },
    });
}

export const Tween = {
    state: [] as number[],
    to: [] as number[],
    duration: [] as number[],
    elapsed: [] as number[],
    easing: [] as number[],
    field: fieldProxy(),
};

traits(Tween, {
    defaults: () => ({
        state: TweenState.Idle,
        to: 0,
        duration: 1,
        elapsed: 0,
        easing: 0,
    }),
    parse: { easing: getEasingIndex },
    enums: { state: TweenState },
});

export function captureFromValue(state: State, tweenEid: number): void {
    const targetEid = state.getFirstRelationTarget(tweenEid, Target);
    const binding = getOrBindAccessor(tweenEid);

    if (binding && targetEid >= 0) {
        fromValues.set(tweenEid, binding.get(targetEid) ?? 0);
    }
}

export function ensureResolved(state: State, tweenEid: number): void {
    const elapsed = Tween.elapsed[tweenEid];
    const duration = Tween.duration[tweenEid];

    if (duration > 0 && elapsed >= duration) return;

    const targetEid = state.getFirstRelationTarget(tweenEid, Target);
    const binding = getOrBindAccessor(tweenEid);

    if (binding && targetEid >= 0) {
        const toValue = Tween.to[tweenEid];
        if (!Number.isFinite(toValue)) {
            throw new Error(`Tween ${tweenEid} has invalid to value: ${toValue}`);
        }
        fromValues.set(tweenEid, binding.get(targetEid) ?? 0);
        binding.set(targetEid, toValue);
    }

    Tween.elapsed[tweenEid] = duration;
}

function updateTweens(state: State, dt: number): void {
    for (const tweenEid of state.query([Tween])) {
        const tweenState = Tween.state[tweenEid];

        if (tweenState === TweenState.Complete) {
            ensureResolved(state, tweenEid);
            continue;
        }

        if (tweenState !== TweenState.Playing) continue;

        const targetEid = state.getFirstRelationTarget(tweenEid, Target);
        const binding = getOrBindAccessor(tweenEid);

        if (Tween.elapsed[tweenEid] === 0 && binding && targetEid >= 0) {
            fromValues.set(tweenEid, binding.get(targetEid) ?? 0);
        }

        Tween.elapsed[tweenEid] += dt;

        const elapsed = Tween.elapsed[tweenEid];
        const duration = Tween.duration[tweenEid];
        const rawProgress = duration <= 0 ? 1 : Math.min(elapsed / duration, 1);

        if (!Number.isFinite(rawProgress)) {
            throw new Error(
                `Tween ${tweenEid} invalid progress: elapsed=${elapsed}, duration=${duration}, dt=${dt}`,
            );
        }

        const easingFn = getEasing(Tween.easing[tweenEid]);
        const easedProgress = easingFn(rawProgress);

        const from = fromValues.get(tweenEid) ?? 0;
        const to = Tween.to[tweenEid];
        const value = from + (to - from) * easedProgress;

        if (!Number.isFinite(value)) {
            throw new Error(
                `Tween ${tweenEid} computed NaN: from=${from}, to=${to}, eased=${easedProgress}, raw=${rawProgress}`,
            );
        }

        if (binding && targetEid >= 0) {
            binding.set(targetEid, value);
        }

        if (rawProgress >= 1) {
            Tween.state[tweenEid] = TweenState.Complete;
        }
    }
}

export interface TweenOptions {
    readonly to: number;
    readonly duration?: number;
    readonly easing?: string;
}

export function createTween(
    state: State,
    targetEid: number,
    fieldPath: string,
    options: TweenOptions,
): number | null {
    const parsed = resolveFieldPath(fieldPath);
    if (!parsed) return null;

    const tweenEid = state.addEntity();
    state.addComponent(tweenEid, Tween);
    Tween.field[tweenEid] = fieldPath;

    const binding = bindFieldAccessor(tweenEid, parsed.component, parsed.field);
    if (!binding) {
        state.removeEntity(tweenEid);
        return null;
    }

    state.addRelation(tweenEid, Target, targetEid);

    Tween.to[tweenEid] = options.to;
    Tween.duration[tweenEid] = options.duration ?? 1;
    Tween.elapsed[tweenEid] = 0;
    Tween.easing[tweenEid] = getEasingIndex(options.easing ?? "linear");

    return tweenEid;
}

export const TweenSystem: System = {
    group: "simulation",

    update(state: State) {
        const dt = state.time.deltaTime;

        resolveAll(state);
        updateSequences(state, dt);
        updateTweens(state, dt);
        checkCompletion(state);
    },
};

export const TweenPlugin: Plugin = {
    name: "Tween",
    systems: [TweenSystem],
    components: { Tween, Sequence, Pause, AutoPlay },
    relations: [Target],
    initialize() {
        fieldPaths.clear();
        accessors.clear();
        fromValues.clear();
        clearDelays();
    },
};
