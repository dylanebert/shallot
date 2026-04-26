---
title: Physics
description: collisions, physics, raycasting
source: standard/physics
icon: atom
---

# Physics

<!-- tabs -->
<!-- tab: UI -->

coming soon

<!-- tab: Code -->

GPU-driven rigid body physics using Augmented Vertex Block Descent (Giles, Diaz, Yuksel — SIGGRAPH 2025). The entire solver runs on the GPU — no CPU readback of body state. ECS Transform for physics bodies is write-only from the GPU side.

Collider types: box, sphere, capsule, hull (arbitrary convex mesh via `hull()`). Joint types: ball and spring. Hard constraints (contacts, ball joints) use augmented Lagrangian. Soft constraints (springs) use penalty only.

## Examples

Add a `Body` component to make an entity physical. Mass, friction, and gravity are set via `Body` fields. Static bodies have mass 0. Gravity defaults to 1 (normal); set to 0 for zero-G.

Joints are declarative components: `BallJoint` for rigid constraints, `SpringJoint` for soft springs. Add them to an entity with `body-a` and `body-b` referencing physics bodies.

`Force` applies persistent linear force and torque to a body. Stays until removed or modified. Fields: `forceX/Y/Z` (Newtons), `torqueX/Y/Z` (Newton-meters). Requires `Body`.

`Impulse` applies a one-shot velocity change. Consumed after one physics frame. Fields: `impulseX/Y/Z` (kg·m/s), `angularImpulseX/Y/Z` (kg·m²/s). Requires `Body`.

`Velocity` sets linear and angular velocity directly. Consumed after one physics frame. Overwrites any Force/Impulse on the same body. Gravity and solver still apply after the override. Fields: `linearX/Y/Z` (m/s), `angularX/Y/Z` (rad/s). Requires `Body`.

`Move` is a marker component for kinematic bodies (mass 0) whose movement should affect touching objects. Without `Move`, repositioning a kinematic body is treated as a teleport — no friction drag, no platform riding. Add `Move` to moving platforms, elevators, and conveyor belts.

Raycasting is available via `raycast()` and shape-specific functions (`raySphere`, `rayOBB`, `rayCapsule`, `rayTriangle`, `rayMesh`).

## Contact Events

The `Contacts` resource provides a double-buffered stream of new collision contacts from the GPU solver. Contacts are emitted for new collisions (warmstart miss) above an impulse threshold. Each contact includes body indices, world-space position, normal, and impulse magnitude.

Create a `ContactReader` cursor and call `readContacts` each frame to iterate new contacts. The callback value is reused — copy fields if you need to store them.

```typescript
import { Contacts, contactReader, readContacts } from "@dylanebert/shallot";

const reader = contactReader();

// in a system update:
const contacts = Contacts.from(state);
if (contacts) {
    readContacts(contacts, reader, (c) => {
        // c.bodyA, c.bodyB — GPU body indices
        // c.posX/Y/Z — world-space contact position
        // c.normalX/Y/Z — contact normal
        // c.impulse — positive impulse magnitude
    });
}
```

Contacts persist for 2 physics ticks. Read every frame to avoid misses. Capacity is 128 contacts per frame — overflow is logged as a warning.


<!-- tab: Reference -->

<!-- API:standard/physics -->

<!-- CORE:physics -->

<!-- /tabs -->
