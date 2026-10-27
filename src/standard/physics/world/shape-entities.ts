import type { EntityRef } from "../../../engine";

/** An ECS shape and body identity a solver shape slot carried at one generation. */
export interface ShapeEntity {
    shape: EntityRef;
    body: EntityRef;
}

interface RetiredShapeEntity extends ShapeEntity {
    generation: number;
}

/** Each solver shape slot's ECS identity at its live generation, in slot-indexed columns, and the
 * identities retired since the last step's events were read, which those events still name. */
export interface ShapeEntities {
    /** The live binding's generation plus one; zero where a slot has none. */
    generations: Float64Array;
    shapes: Float64Array;
    bodies: Float64Array;
    retired: Map<number, RetiredShapeEntity[]>;
}

export function createShapeEntities(): ShapeEntities {
    return {
        generations: new Float64Array(64),
        shapes: new Float64Array(64),
        bodies: new Float64Array(64),
        retired: new Map(),
    };
}

/** Binds `slot` at `generation` to its entities, retiring a live binding of another generation. */
export function bindShapeEntity(
    entities: ShapeEntities,
    slot: number,
    generation: number,
    shape: EntityRef,
    body: EntityRef,
): void {
    if (slot >= entities.generations.length) {
        let capacity = entities.generations.length;
        while (capacity <= slot) capacity *= 2;
        for (const column of ["generations", "shapes", "bodies"] as const) {
            const grown = new Float64Array(capacity);
            grown.set(entities[column]);
            entities[column] = grown;
        }
    }
    const live = entities.generations[slot];
    if (live !== 0 && live !== generation + 1) retireShapeEntity(entities, slot, live - 1);
    entities.generations[slot] = generation + 1;
    entities.shapes[slot] = shape;
    entities.bodies[slot] = body;
}

/** Retires `slot`'s live binding when it is at `generation`. */
export function retireShapeEntity(entities: ShapeEntities, slot: number, generation: number): void {
    if (slot >= entities.generations.length || entities.generations[slot] !== generation + 1)
        return;
    let retired = entities.retired.get(slot);
    if (!retired) entities.retired.set(slot, (retired = []));
    retired.push({
        generation,
        shape: entities.shapes[slot] as EntityRef,
        body: entities.bodies[slot] as EntityRef,
    });
    entities.generations[slot] = 0;
}

/** Copies the live binding of `slot` into `out`, if any. */
export function liveShapeEntity(entities: ShapeEntities, slot: number, out: ShapeEntity): boolean {
    if (slot < 0 || slot >= entities.generations.length || entities.generations[slot] === 0)
        return false;
    out.shape = entities.shapes[slot] as EntityRef;
    out.body = entities.bodies[slot] as EntityRef;
    return true;
}

/** The slot's live generation, or -1 when it has no live binding. */
export function liveShapeGeneration(entities: ShapeEntities, slot: number): number {
    return slot >= 0 && slot < entities.generations.length ? entities.generations[slot] - 1 : -1;
}

/** Copies the binding `slot` had at `generation`, live or retired since the last prune, into `out`. */
export function shapeEntityAt(
    entities: ShapeEntities,
    slot: number,
    generation: number,
    out: ShapeEntity,
): boolean {
    if (liveShapeGeneration(entities, slot) === generation)
        return liveShapeEntity(entities, slot, out);
    const retired = entities.retired.get(slot);
    if (!retired) return false;
    for (let i = retired.length - 1; i >= 0; --i)
        if (retired[i].generation === generation) {
            out.shape = retired[i].shape;
            out.body = retired[i].body;
            return true;
        }
    return false;
}

/** Drops retired bindings once their step's events are read. */
export function pruneShapeEntities(entities: ShapeEntities): void {
    if (entities.retired.size !== 0) entities.retired.clear();
}

export function clearShapeEntities(entities: ShapeEntities): void {
    entities.generations.fill(0);
    entities.retired.clear();
}
