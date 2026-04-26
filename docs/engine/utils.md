---
title: Utils
description: math helpers, vectors, shapes
source: engine/utils
icon: calculator
order: 3
---

# Utils

<!-- tabs -->
<!-- tab: UI -->

coming soon

<!-- tab: Code -->

Math functions and shape constants shared across the engine. All functions operate on plain numbers and `Float32Array`s — no special vector or matrix classes.

## Interpolation

```typescript
clamp(value, min, max)   // constrain to range
lerp(a, b, t)            // linear interpolation (t: 0→a, 1→b)
```

`slerp` does spherical interpolation between two quaternions:

```typescript
const q = slerp(fromX, fromY, fromZ, fromW, toX, toY, toZ, toW, t);
// returns { x, y, z, w }
```

## Rotations

Quaternions internally, Euler angles (degrees) in scene files and most gameplay code.

```typescript
eulerToQuaternion(0, 45, 0)                    // degrees → quaternion { x, y, z, w }
quaternionToEuler(x, y, z, w)                  // quaternion → degrees { x, y, z }
rotateQuaternion(qx, qy, qz, qw, dx, dy, dz)  // apply degree delta to quaternion
```

### Look At

```typescript
lookAt(eyeX, eyeY, eyeZ, targetX, targetY, targetZ)  // → quaternion { x, y, z, w }
lookAtMatrix(eyeX, eyeY, eyeZ, targetX, targetY, targetZ)  // → Float32Array(16) view matrix
```

Both accept optional up vector args (default: 0, 1, 0).

## Projection

```typescript
perspective(fov, aspect, near, far)    // fov in degrees → Float32Array(16)
orthographic(size, aspect, near, far)  // → Float32Array(16)
orthographicBounds(left, right, bottom, top, near, far)  // explicit bounds
```

All accept an optional `out: Float32Array` to avoid allocation.

## Matrix Operations

```typescript
multiply(a, b)       // 4×4 matrix multiplication
invert(m)            // fast inverse — rotation + translation only (no scale/projection)
invertMatrix(m)      // general 4×4 inverse
```

Use `invert` when you know the matrix is rigid (view matrices). Use `invertMatrix` for projection or scaled matrices.

## Frustum and Collision

```typescript
extractFrustumPlanes(viewProj)        // → Float32Array(24), 6 normalized planes
extractFrustumCorners(invViewProj, nearZ, farZ)  // → Float32Array(24), 8 corner points

testAABBFrustum(minX, minY, minZ, maxX, maxY, maxZ, planes)  // → boolean
testAABBSphere(minX, minY, minZ, maxX, maxY, maxZ, cx, cy, cz, radius)  // → boolean
```

## Shapes

Shape constants map to GPU primitive indices:

```typescript
Shape.Box      // 0
Shape.Sphere   // 1
Shape.Capsule  // 2
Shape.Plane    // 3
Shape.Mesh     // 255
```

`shapeToPrimitive(shape)` converts to the GPU index.

## Types

```typescript
interface Vec3 { x: number; y: number; z: number }
interface Ray { origin: Vec3; direction: Vec3 }
```


<!-- tab: Reference -->

<!-- API:engine/utils -->

<!-- /tabs -->
