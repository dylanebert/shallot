---
title: Render
description: drawing objects, materials, shadows
source: standard/render
icon: brush
---

# Render

<!-- tabs -->
<!-- tab: UI -->

coming soon

<!-- tab: Code -->

Shallot's rendering runs as compute graph nodes. The contract: geometry, surface (shader), properties (per-entity data for surfaces).

## Mesh format

Vertex stride is 8 floats: position (3) + normal (3) + UV (2). All `MeshData.vertices` arrays must follow this layout. Wrong stride silently corrupts geometry.

## Surfaces

Surfaces are WGSL shader snippets (vertex + fragment) that define how Parts render. No `discard` in surface fragments — set `(*surface).opacity = 0.0;` instead.

Surface properties (`properties` field) declare per-entity data accessible as `inst.fieldName` in WGSL. Use `property("fieldName")` to get the backing typed array for CPU writes.

## Registries

Meshes and surfaces use `Registry<T>` — register with `mesh(data, name?)` / `surface(data, name?)`, query with `meshRegistry` / `surfaceRegistry`.

## Dynamic meshes

`Dynamic` tag component. Entities with `Dynamic` get a per-instance vertex buffer clone in the shape atlas. All updates are GPU-driven — no per-frame CPU upload.

## Pipeline extension

Custom rendering pipelines import from `@dylanebert/shallot/render/core`. This provides WGSL struct definitions, shader utilities, surface compilation, batching, culling, and pass infrastructure. The engine's own raster and raytracing pipelines use this same API.


<!-- tab: Reference -->

<!-- API:standard/render -->

<!-- CORE:render -->

<!-- /tabs -->
