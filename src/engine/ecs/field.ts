import type { SchemaField, Type } from "./component";
import { fieldSchema } from "./component";

/** Declare a typed component field; its world-owned column grows with its highest eid. */
export function field<T extends Type & { readonly lanes: 1 }>(type: T): SchemaField<T>;
export function field<T extends Type & { readonly lanes: 2 }>(type: T): SchemaField<T>;
export function field<T extends Type & { readonly lanes: 4 }>(type: T): SchemaField<T>;
export function field(type: Type): SchemaField<Type> {
    return fieldSchema(type);
}
