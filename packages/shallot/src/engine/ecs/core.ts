// #doc:dev
// ### Reflection: reading the world without knowing its shape
//
// The editor never imports a game's components. It draws its inspector, outliner, and add-component
// picker entirely from reflection over the live registry — the same surface any custom tool (a debugger, a
// save system, a level exporter) builds on.
//
// `entries()` walks every registered component; `schema(name)` gives one its field layout — type, default,
// kind — without holding the component object; `inspect(state, eid)`, `find`, and `snapshot` read live
// values; `dependencies` / `exclusions` / `provides` / `isSingleton` surface the traits the editor draws as
// chips. Registration keys by a stable id interned by name, so a component's data survives a module reload —
// see the ECS contract for the reload rules.

export { type Alias, eulerAlias, laneAlias } from "../utils";
export { fields, idOf, lanes, refs } from "./component";
export {
    camel,
    dependencies,
    dump,
    type EntityData,
    exclusions,
    type FieldInfo,
    type FieldKind,
    type FieldValues,
    find,
    inspect,
    isSingleton,
    kebab,
    provides,
    readFields,
    type Schema,
    schema,
    snapshot,
} from "./reflection";
export {
    clear,
    entries,
    getComponent,
    getExclusions,
    getTraits,
    register,
    type Traits,
} from "./traits";
