---
title: ECS
description: entities, components, systems
source: engine/ecs
icon: boxes
order: 0
---

# ECS

<!-- tabs -->
<!-- tab: UI -->

coming soon

<!-- tab: Code -->

## Entities

An entity is a number.

```typescript
const eid = state.addEntity();
```

## Components

A component is data.

```typescript
const Health = {
    current: [] as number[],
    max: [] as number[],
};
```

Add it to an entity to give that entity health:

```typescript
state.addComponent(eid, Health);
Health.current[eid] = 50;
Health.max[eid] = 100;
```

## Systems

A system runs code every frame.

```typescript
const DamageSystem: System = {
    update(state) {
        Health.current[eid] -= 1;
    },
};
```

This system reduces health by 1 for the entity `eid`.

## Queries

A query finds all entities with specific components.

```typescript
const DamageSystem: System = {
    update(state) {
        for (const eid of state.query([Health])) {
            Health.current[eid] -= 1;
        }
    },
};
```

This system reduces health by 1 for every entity with the `Health` component.

## Plugins

A plugin bundles components and systems together.

```typescript
const HealthPlugin: Plugin = {
    name: "Health",
    components: { Health },
    systems: [DamageSystem],
};
```

## Relations

A relation links one entity to another. `ChildOf` makes an entity a child of a parent.

```typescript
state.addRelation(child, ChildOf, parent);
```

## Resources

Resources are global data that don't belong to an entity.

```typescript
const Volume = resource<number>("volume");
state.setResource(Volume, 0.8);
state.getResource(Volume); // 0.8
```

## Events

Events are frame-scoped queues for cross-system messages. Drained automatically at end-of-step.

```typescript
const Damage = events<{ amount: number }>("damage");
Damage.send(state, { amount: 10 });
for (const ev of Damage.read(state)) console.log(ev.amount);
```

Use events for transient cross-system signals where the receiver isn't tied to a specific entity (placement, hit, level-up). Use marker components for entity-scoped actions, resources for persistent state.

## Hooks

Run code when a component is added or removed.

```typescript
state.observe(onAdd(Health), (eid) => {
    Health.current[eid] = Health.max[eid];
});
```

<!-- tab: Reference -->

<!-- API:engine/ecs -->

<!-- CORE:ecs -->

<!-- /tabs -->
