---
paths:
    - "packages/shallot/src/engine/**/*.ts"
    - "packages/shallot/src/standard/**/*.ts"
    - "packages/shallot/src/extras/**/*.ts"
    - "examples/**/*.scene"
    - "examples/**/*.ts"
---

# ECS

## Plugin lifecycle

Initialize registers pre-scene; warm handles GPU/scene data/derived spawns; setup is lazy per-State, update pure over State, dispose teardown. Phases are idempotent/re-runnable. Mesh/surface producers depend on RenderPlugin after its wipe. Build rejects ALL missing dependencies before GPU/lifecycle/registry effects; optional peers use conditional composition/nullable hooks.

## Reload-safety: lifecycle + module scope

Module scope holds definitions/registries, not eids, entity handles or accumulators; derive time from State. Initialize clears/rebuilds registries, removing toggled-off producers. Global handles need identity-guarded teardown against stale States. Per-build cleanup lives beside creation: `state.onDispose` (LIFO) or `state.signal`; plugin dispose is module/process lifetime. UI needs real unmount/loop cancellation, not host removal. Re-warm clears prior mounts first AND retains State cleanup.

Capacity, component set/schema and membership generations require fresh builds to change. Identity is name-interned, Symbol-keyed, non-enumerable; re-registration preserves stores/GPU buffers. Anonymous identities are object-stable; identity work stays slow-path.

## Runtime state

Serialize on demand via shared `readComponent`, never per-frame/duplicate codecs: authored values/identity, not GPU buffers/warm spawns. Explicit eid sets admit procedural entities. Marker + `not()` gates one-time work. Process singletons: matching PascalCase type/value, plugin-initialized, direct reads, no setters/wrappers.

## Component storage

Sparse is CPU, slab GPU-read; both expose Single/Pair/Quad. Typed arrays, not number arrays; one Pair/Quad per vector, no split-suffix support. Pick narrow accurate types; hex stays f32. Packed mirrors keep CPU lossless, packing only GPU flushes. Dirty bits stay per-field/per-entity, set EVERY write, cleared ONLY at frame flush, never merged/cleared early: future delta-reader contract.

Remove/destroy clears membership, NOT data; add reapplies defaults. GPU index scans MUST gate membership, not sentinels/field clearing. Hoist once, mutate, write back; lone accesses stay direct. Shared codec owns array/dotted defaults/lane attributes. Fractional f32 tests use closeness, exact only if representable. Euler/hex/matrix conveniences stay in aliases/codecs. Import storage via engine barrel. Patterns: `standard/transforms/index.ts`, `engine/scene/xml.test.ts`.

## Entity reference fields

Refs use entity type, not u32/parallel lists; scenes use `@name`, never eids. Serialize mints names inside its set, uses recorded names outside, throws for destroyed/unnamed external targets. Flat scenes/world Transform: no nesting/parent graph/cascade destroy. Hierarchy/animation producers emit flat output; palettes/VAT bind separately.

An eid is a borrow: bare index, never packed version. Re-query; held refs need BOTH membership and matching `state.stamp(eid)` against despawn/reuse. Stamps increment on allocation, not destroy; zero means never created, eid zero reserved, capacity admits capacity−1. Keep stamps beside held state; extract pure diffs, not universal adoption wrappers. Serialization identity is separate.

## Anti-patterns

No last-State/exists guards, camera skip-checks or module ownership maps: scope state/use relations/markers. Post-warm GPU buffers must exist: read directly, throw if missing, cache groups until identity changes, never gather-and-skip.

## Bevy as the structural reference

Take data-first ECS/plugins, named resource publish/subscribe, closed typed resource unions and kind tags. Skip separate render/extract apps, asset refcounts, open reflection unions, manual slot edges (auto-wire names), macro/query/ordering DSLs, parallel/conflict executors, change ticks, packed generations, hidden deferred sync/mutations and typed system-param chains. One State: sim/render, immediate mutation, camera/canvas views, after BeginFrameSystem, compile in warm; no universal compute graph/compositor.

Adopt for real needs/compounding iteration and speed, without hidden behavior/authoring tax. Stable substrate earns strict pipeline types; changing substrate stays loose. No forced analogues against WebGPU/procedural-first constraints; the small scheduler is deliberate.
