---
title: Scene
description: building worlds with scene files
source: engine/scene
icon: file-code
order: 1
---

# Scene

<!-- tabs -->
<!-- tab: UI -->

coming soon

<!-- tab: Code -->

Scene files describe the world in XML. Each element creates an entity. Each attribute adds a component.

```xml
<scene>
    <a id="camera" camera orbit transform />
    <a ambient-light />
    <a directional-light />
    <a id="ground" transform="pos: 0 -1 0" part="shape: plane; scale: 10" />
</scene>
```

## Attributes

A bare attribute adds the component with defaults:

```xml
<a camera />
<a transform />
```

Set fields with `field: value` pairs separated by semicolons:

```xml
<a part="shape: box; scale: 1 2 1; color: #ff8800" />
<a transform="pos: 0 5 0; rot: 0 45 0; scale: 2" />
```

Field names are kebab-case in scene files, camelCase in code.

### Values

- Numbers: `5`, `3.14`, `-1`
- Booleans: `true`, `false`
- Hex colors: `#ff8800`, `0xff8800`
- Enums: string names like `auto`, `loop` (component-dependent)
- Entity refs: `@id` (see below)

### Vectors

Vector fields accept all components or a single broadcast value:

```xml
<a transform="pos: 0 5 0" />   <!-- x=0, y=5, z=0 -->
<a transform="scale: 2" />     <!-- x=2, y=2, z=2 -->
```

Works for vec2, vec3, and vec4. Detected automatically from component field names (`X`/`Y`, `X`/`Y`/`Z`, `X`/`Y`/`Z`/`W`).

## Entity References

`@` references another entity by its `id`:

```xml
<a id="target" transform />
<a tween="target: @target; field: pos-y; to: 5" />
```

Works for both relation attributes and entity ID fields.

## Nesting

Child elements get a `ChildOf` relation to their parent, driving the transform hierarchy:

```xml
<a id="arm" transform>
    <a id="hand" transform="pos: 0 -1 0" part="shape: sphere" />
</a>
```

## Parsing and Loading

`parse()` converts XML to a node tree. `load()` creates entities in ECS state.

```typescript
const nodes = parse(xml);
const nodeToEntity = load(nodes, state); // Map<Node, number>
```

You can inspect or transform nodes between `parse()` and `load()`. Usually you don't call these directly — `run()` handles it.

## Diagnostics

`diagnose()` validates a parsed scene against registered components:

```typescript
for (const d of diagnose(parse(xml))) {
    console.warn(d.message);
    // '"transfrom" is not registered, did you mean "transform"?'
}
```

Returns `Diagnostic[]` with `node`, `attr`, `kind` (`"unregistered"` | `"missing-requires"`), and `message`.

## Serialization

Round-trip between string and structured form:

```typescript
const fields = parseFields("transform", "pos: 0 5 0; rot: 0 45 0");
// { posX: 0, posY: 5, posZ: 0, rotX: 0, rotY: 45, rotZ: 0 }

formatFields("transform", fields);
// "pos: 0 5 0; rot: 0 45 0"
```

`formatFields` collapses vectors and strips defaults. Pass `{ stripDefaults: false }` to keep them.

`normalizeAttr(name, value)` round-trips for canonical form. Returns `null` on invalid input.

## Field Access

Write component fields programmatically:

```typescript
setFieldValue(Transform, "posY", eid, 5);
setString(Text, "content", eid, "Hello");
isStringField(component, "content"); // true if string-backed
```

## Node Types

```typescript
interface Node {
    id?: string;
    attrs: Attr[];
    children: Node[];
}

interface Attr {
    name: string;
    value: string;
}
```

`serialize(nodes)` converts back to XML. `findNodeById(nodes, id)` searches by ID. `findParent(nodes, node)` finds a node's parent.


<!-- tab: Reference -->

<!-- API:engine/scene -->

<!-- /tabs -->
