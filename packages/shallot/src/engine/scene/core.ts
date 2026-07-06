// #doc:dev
// ### The codec: scene text ↔ live components
//
// A scene file is the author's source; the editor and any scene tooling need to move a value across the
// text↔state boundary in both directions, per attribute. That codec is the same surface the inspector, the
// document model, and the scene formatter all build on — and it's here for a custom inspector, a
// scene-transform script, or a level exporter to build on too.
//
// One attribute round-trips through four functions. `parseFields(name, value)` turns an attribute string
// (`"pos: 0 5 0"`) into a field record; `setFieldValue(component, field, eid, value)` writes one field
// into a live component; `readComponent(name, component, eid)` reads the entity's live values back to a
// scene string; `formatFields(name, record)` renders a field record to canonical text (vectors collapsed,
// defaults stripped). `normalizeAttr(name, value)` is parse-then-format in one call — the scene formatter
// runs every `.scene` through it so a hand-authored file settles to the exact bytes the editor's save path
// emits, one canonical form with no divergence.
//
// `findNodeById` and `findParent` navigate a parsed `Node` tree (from `parse`) before it's loaded — the
// editor's outliner and drag-reparent are built on them.
//
// `Preloads` is the pre-load resolve seam: a plugin whose assets scenes reference by name (the glTF
// importer) registers a `Preloader`, and `preload(nodes, state)` — awaited between `parse` and `load` by
// the engine and the editor — lets it import what the scene names before any name resolves.

export {
    formatFields,
    normalizeAttr,
    parseFields,
    readComponent,
    setFieldValue,
} from "./codec";
export { type Preloader, Preloads, preload } from "./preload";
export { findNodeById, findParent } from "./xml";
