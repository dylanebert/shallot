import { type Plugin, type State, State as StateClass } from "../../engine";
import { clear, register } from "../../engine/ecs/traits";
import { Slab } from "../slab";
import { Body, bodyTraits, Joint, jointTraits, PhysicsPlugin, Spring, springTraits } from "./index";

// Shared build-up for the physics checks: a headless `State` carrying `PhysicsPlugin`, with no GPU
// device and no `app()`. Not a check itself and never published (`*.fixture.ts` is excluded from the
// package), so it holds only what a check would otherwise repeat.

/** wire a plugin's systems into state without going through `app({ plugins })`. */
export function attach(state: State, plugin: Plugin): void {
    for (const s of plugin.systems ?? []) state.addSystem(s, plugin.name);
}

/**
 * Build a headless `State` with the physics plugin warmed: components registered, slab collected,
 * world created, systems attached. The caller authors `Body`/`Spring`/`Joint` entities and steps.
 */
export async function headlessPhysicsState(): Promise<State> {
    clear();
    const state = new StateClass();
    register("body", Body, bodyTraits);
    register("spring", Spring, springTraits);
    register("joint", Joint, jointTraits);
    Slab.collect();
    PhysicsPlugin.initialize?.(state);
    await PhysicsPlugin.warm?.(state);
    attach(state, PhysicsPlugin);
    return state;
}

/** author one `Body` entity on a headless state. */
export function addBody(
    state: State,
    body: {
        shape: number;
        pos: [number, number, number];
        halfExtents: [number, number, number, number];
        mass: number;
        friction?: number;
        quat?: [number, number, number, number];
    },
): number {
    const eid = state.create();
    state.add(eid, Body);
    Body.shape.set(eid, body.shape);
    Body.halfExtents.set(eid, ...body.halfExtents);
    Body.pos.set(eid, body.pos[0], body.pos[1], body.pos[2], 0);
    Body.quat.set(eid, ...(body.quat ?? [0, 0, 0, 1]));
    Body.mass.set(eid, body.mass);
    Body.friction.set(eid, body.friction ?? 0.5);
    return eid;
}
