export {
    createColorProxy,
    createFieldProxy,
    getComponent,
    getComponentName,
    getComponents,
    getFieldLayout,
    getTraits,
    isStringField,
    registerComponent,
    clearRegistry,
    type ComponentEntry,
    type FieldLayout,
    type FieldProxy,
    type Derived,
    type Traits,
} from "./component";

export type { ArrayKind } from "./capacity";

export { formatHex, toKebabCase, toCamelCase } from "./strings";

export { toposort, CycleError } from "./scheduler";

export { registerRelation, getRelation, type Relation } from "./relation";

export {
    schema,
    schemas,
    dependencies,
    inspect,
    find,
    snapshot,
    dump,
    readFields,
    detectVec2,
    detectVec3,
    detectVec4,
    type Schema,
    type FieldInfo,
    type FieldKind,
    type FieldValues,
    type EntityData,
} from "./reflection";
