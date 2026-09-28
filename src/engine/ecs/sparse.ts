import type { SchemaField, Type } from "./component";
import { fieldSchema } from "./component";

/**
 * Declare a typed component field whose world-owned column is demand-grown. The field descriptor is a
 * schema only: systems read and write through `state.of(Component)`, never through this shared object.
 *
 * @example
 * const Orbit = { yaw: sparse(f32), pan: sparse(vec2) };
 * const orbit = state.of(Orbit);
 * orbit.yaw.set(eid, 1.2);
 */
export function sparse<T extends Type & { readonly lanes: 1 }>(type: T): SchemaField<T>;
export function sparse<T extends Type & { readonly lanes: 2 }>(type: T): SchemaField<T>;
export function sparse<T extends Type & { readonly lanes: 4 }>(type: T): SchemaField<T>;
export function sparse(type: Type): SchemaField<Type> {
    return fieldSchema(type, "sparse");
}
