import type { Entity } from "./entity";

/**
 * per-world named entity identity and authored membership. Dropped on destroy so recycled
 * entities do not inherit names. Holds no serialization logic, just the map.
 */
export class Identity {
    private _ids = new Map<Entity, string>();
    private _authored = new Set<Entity>();

    /** mark an entity as authored, recording its id when named */
    author(eid: Entity, id?: string): void {
        this._authored.add(eid);
        if (id !== undefined) this._ids.set(eid, id);
        else this._ids.delete(eid);
    }

    /** the scene `id` an entity was named with, or undefined if anonymous / not authored */
    id(eid: Entity): string | undefined {
        return this._ids.get(eid);
    }

    /** the load-authored entities, in author order (warm-derived entities are absent by construction) */
    get authored(): ReadonlySet<Entity> {
        return this._authored;
    }

    /** drop an entity's identity — called on `destroy` so a recycled eid never inherits a stale id */
    forget(eid: Entity): void {
        this._ids.delete(eid);
        this._authored.delete(eid);
    }
}
