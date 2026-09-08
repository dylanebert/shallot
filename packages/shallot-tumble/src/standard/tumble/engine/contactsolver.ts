// The step/solve context plus the TS side of the contact solve — the narrowphase ↔ kernel bridge.
//
// The contact solve itself (prepare / warm-start / solve / relax / restitution / store) runs in the
// wasm kernel — the scalar path (kernel/src/contact.rs) for mesh + overflow contacts, the wide 4-lane
// path (kernel/src/contact_wide.rs) for convex contacts in a real graph color. The manifolds are
// column-resident (the persistent store, manifoldstore.ts), so the solver gathers each contact's
// material + manifold straight out of the store keyed by contactId — no per-step marshal of the
// manifold data. This module owns the TS half: the `StepContext` threaded through the whole solve, and
// the per-step slot index the kernel gathers through.
//
// Every touching contact takes one solver record, ordered [convex | mesh | overflow], each region
// walked color-by-color (ascending color index). The wide path maps each record's four lanes to their
// contactIds; the scalar path maps each record to its contactId + its slice of the transient
// constraint columns. `computeLayout` derives the counts + per-color spans (so the transient columns
// can be sized); `writeSlots` fills the slot index + each contact's per-step directory row; the kernel
// `store` writes the solved impulses straight back into the pool, and `readbackHitEvents` collects the
// contacts it flagged.

import type { BodySim, BodyState } from "./body";
import { COLOR_SPAN_STRIDE, type Columns, SLOT_STRIDE, WIDE_META_STRIDE } from "./columns";
import { OVERFLOW_INDEX } from "./core";
import type { GraphColor } from "./graph";
import type { Softness } from "./softness";
import type { WorldState } from "./world";

/** SIMD lane width (B3_SIMD_WIDTH): convex contacts pack 4 to a wide record. */
const LANES = 4;

/** The per-step solver context threaded through the solve (b3StepContext, scalar subset). */
export type StepContext = {
    world: WorldState;
    sims: BodySim[];
    states: BodyState[];
    dt: number;
    invDt: number;
    h: number;
    invH: number;
    subStepCount: number;
    contactSoftness: Softness;
    staticSoftness: Softness;
    restitutionThreshold: number;
    maxLinearVelocity: number;
    enableWarmStarting: boolean;
    // Sleep bookkeeping filled by finalize (the C per-worker b3TaskContext fields). awakeIslands is
    // indexed by an awake island's localIndex; the split candidate is the sleepiest split-pending
    // island, reduced into world.splitIslandId for the next step.
    awakeIslands: boolean[];
    splitIslandId: number;
    splitSleepTime: number;
    // Fast bullet body sims collected during finalize; swept in the deferred bullet stage (CCD).
    bulletBodies: BodySim[];
    // Contact ids flagged for a hit event during impulse store, and joint ids flagged over their
    // force/torque threshold during the biased joint solve. Both are walked in ascending id order
    // after the solve to build the user-facing event arrays (b3's per-worker bit sets, serial).
    hitEventContacts: Set<number>;
    jointEventFlags: Set<number>;
};

/** One active graph color's constraint ranges for the per-color solve interleave. `wide*` index the
 * wide-record range (each record groups up to 4 convex contacts); `mesh*` index the scalar
 * contact-record range in the input pool. `color` carries the color's joints (solved TS-side). */
export type ColorSpan = {
    color: GraphColor;
    wideStart: number;
    wideCount: number;
    meshStart: number;
    meshCount: number;
};

/** The per-step contact solve layout — counts to size the columns, plus the ranges the solve drives.
 * `colors` is the active graph colors (ascending); `mesh`/`wide` are the flat regions prepare/store
 * run over; `overflow` is the serial spill range. */
export type SolveLayout = {
    contacts: number;
    manifolds: number;
    points: number;
    wide: number;
    colors: ColorSpan[];
    meshStart: number;
    meshTotal: number;
    wideTotal: number;
    overflowStart: number;
    overflowCount: number;
};

/** Reused out-record for `contactExtent` — read `.manifolds`/`.points` immediately after the call,
 * before the next `contactExtent`. Kills the per-contact object churn (~2 per contact per step). */
const extent = { manifolds: 0, points: 0 };

/** Fills `extent` with the total manifold + point count of a contact's current narrowphase output. */
function contactExtent(world: WorldState, contactId: number): void {
    const contact = world.contacts[contactId];
    const manifolds = contact.manifoldCount;
    let points = 0;
    for (let m = 0; m < manifolds; ++m) {
        points += contact.manifolds[m].pointCount;
    }
    extent.manifolds = manifolds;
    extent.points = points;
}

// Reused per-step layout scratch. Single-live-world + synchronous stepping (one computeLayout call per
// step, its result consumed before the next), so module scratch is safe — the same pattern as
// `awakeIslandsScratch` in solver.ts. `layoutScratch.colors` is `spansScratch` permanently; every field
// of every used slot is rewritten each call and the array is truncated to the active-color count, so no
// stale span is ever observable (the truncation means spans re-mint if the active-color count
// oscillates — stable in steady state, not strictly grow-only). `spansScratch` holds live GraphColor
// refs, but they're overwritten before any read each step (the pooled objects never outlive the step's
// consumers).
const activeScratch: number[] = [];
const spansScratch: ColorSpan[] = [];
const layoutScratch: SolveLayout = {
    contacts: 0,
    manifolds: 0,
    points: 0,
    wide: 0,
    colors: spansScratch,
    meshStart: 0,
    meshTotal: 0,
    wideTotal: 0,
    overflowStart: 0,
    overflowCount: 0,
};

/**
 * Derive the contact solve layout for this step: the column sizes and the per-color / flat ranges the
 * kernel phases run over. Walks the constraint graph once; touches no columns (called before they are
 * reserved). Pool order is [convex | mesh | overflow], each region walked in ascending color index.
 * Reuses module scratch (the returned layout + its `colors` are pooled — consumed synchronously
 * within the same step, never retained across steps).
 */
export function computeLayout(world: WorldState): SolveLayout {
    const colors = world.constraintGraph.colors;

    // Active colors (occupancy > 0), ascending. The overflow color is handled separately.
    const active = activeScratch;
    active.length = 0;
    for (let i = 0; i < OVERFLOW_INDEX; ++i) {
        const c = colors[i];
        if (c.convexContacts.length + c.contacts.length + c.jointSims.length > 0) {
            active.push(i);
        }
    }

    // First sub-region: convex contacts, color by color. Contact record + wide record cursors.
    let contactCursor = 0;
    let manifoldCursor = 0;
    let pointCursor = 0;
    let wideCursor = 0;
    const spans = spansScratch;
    for (let a = 0; a < active.length; ++a) {
        const color = colors[active[a]];
        const nConvex = color.convexContacts.length;
        const wideStart = wideCursor;
        for (let j = 0; j < nConvex; ++j) {
            contactExtent(world, color.convexContacts[j]);
            manifoldCursor += extent.manifolds;
            pointCursor += extent.points;
            contactCursor += 1;
        }
        const wideCount = nConvex > 0 ? ((nConvex - 1) >> 2) + 1 : 0;
        wideCursor += wideCount;
        // Reuse the pooled span, or grow the pool once; every field is written unconditionally.
        let span = spans[a];
        if (span === undefined) {
            span = { color, wideStart, wideCount, meshStart: 0, meshCount: 0 };
            spans[a] = span;
        } else {
            span.color = color;
            span.wideStart = wideStart;
            span.wideCount = wideCount;
            span.meshStart = 0;
            span.meshCount = 0;
        }
    }
    // Drop any pooled spans past this step's active-color count so no stale span is observable.
    spans.length = active.length;

    // Second sub-region: mesh contacts, color by color.
    const meshStart = contactCursor;
    for (let a = 0; a < active.length; ++a) {
        const color = colors[active[a]];
        const nMesh = color.contacts.length;
        spans[a].meshStart = contactCursor;
        spans[a].meshCount = nMesh;
        for (let k = 0; k < nMesh; ++k) {
            contactExtent(world, color.contacts[k].contactId);
            manifoldCursor += extent.manifolds;
            pointCursor += extent.points;
            contactCursor += 1;
        }
    }
    const meshTotal = contactCursor - meshStart;

    // Third sub-region: the overflow spill (serial, scalar).
    const overflow = colors[OVERFLOW_INDEX];
    const overflowStart = contactCursor;
    const overflowCount = overflow.contacts.length;
    for (let k = 0; k < overflowCount; ++k) {
        contactExtent(world, overflow.contacts[k].contactId);
        manifoldCursor += extent.manifolds;
        pointCursor += extent.points;
        contactCursor += 1;
    }

    const layout = layoutScratch;
    layout.contacts = contactCursor;
    layout.manifolds = manifoldCursor;
    layout.points = pointCursor;
    layout.wide = wideCursor;
    layout.meshStart = meshStart;
    layout.meshTotal = meshTotal;
    layout.wideTotal = wideCursor;
    layout.overflowStart = overflowStart;
    layout.overflowCount = overflowCount;
    return layout;
}

/** Write the active colors' spans into the `colorSpan` column for the batched (jointless) color loop:
 * per color, wideStart, wideCount, meshStart, meshCount (the same ranges the per-color TS path drives). */
export function writeColorSpans(cols: Columns, layout: SolveLayout): void {
    const span = cols.colorSpan;
    for (let i = 0; i < layout.colors.length; ++i) {
        const c = layout.colors[i];
        const o = i * COLOR_SPAN_STRIDE;
        span[o] = c.wideStart;
        span[o + 1] = c.wideCount;
        span[o + 2] = c.meshStart;
        span[o + 3] = c.meshCount;
        // Jointless path: no colored joints. The staged solve reads these; zero them so a reused
        // color-span column never feeds stale joint spans into a later jointed build.
        span[o + 4] = 0;
        span[o + 5] = 0;
    }
}

/** Write one contact's per-step directory row (the material + body sim indices the kernel gathers).
 * `NULL_INDEX` (-1) body indices land as `0xFFFFFFFF` on the u32 write (= the kernel's `NULL_INDEX`). */
function writeRow(world: WorldState, contactId: number): void {
    const contact = world.contacts[contactId];
    world.manifoldStore.writeContactRow(
        contactId,
        contact.friction,
        contact.restitution,
        contact.rollingResistance,
        contact.tangentVelocity,
        contact.flags,
        contact.bodySimIndexA,
        contact.bodySimIndexB,
    );
}

/**
 * Write every touching contact's per-step directory row + the solver slot index the kernel gathers
 * through, following the record order `computeLayout` fixed ([convex | mesh | overflow], each region
 * color-by-color). The convex region fills the wide records' lane map (contactId per lane + active
 * lane count); the mesh + overflow regions fill the scalar slot column (contactId + the transient
 * `mc`/`mcp` bases). The manifold data itself is already column-resident (the narrowphase wrote it),
 * so nothing copies it — the kernel reads it straight from the pool.
 */
export function writeSlots(cols: Columns, world: WorldState, layout: SolveLayout): void {
    const slot = cols.slotScalar;
    const wideMeta = cols.wideMeta;
    // Transient constraint-column cursors: the mesh/overflow records' `mc`/`mcp` bases start past the
    // convex manifolds, so the convex region advances them too (matching computeLayout's cursor).
    let gm = 0;
    let gp = 0;
    let c = 0;

    // Convex region: directory row + the per-record lane map. Records never cross a color boundary;
    // a color's tail record is short.
    for (const span of layout.colors) {
        const convex = span.color.convexContacts;
        const nConvex = convex.length;
        for (let j = 0; j < nConvex; ++j) {
            const contactId = convex[j];
            writeRow(world, contactId);
            const rec = span.wideStart + (j >> 2);
            wideMeta[rec * WIDE_META_STRIDE + (j & 3)] = contactId;
            contactExtent(world, contactId);
            gm += extent.manifolds;
            gp += extent.points;
            ++c;
        }
        for (let r = 0; r < span.wideCount; ++r) {
            const rec = span.wideStart + r;
            const lanes = nConvex - r * LANES;
            wideMeta[rec * WIDE_META_STRIDE + LANES] = lanes < LANES ? lanes : LANES;
        }
    }

    // Mesh region, color by color (contiguous — matches layout.meshStart / meshTotal).
    for (const span of layout.colors) {
        const contacts = span.color.contacts;
        for (let k = 0; k < contacts.length; ++k) {
            const contactId = contacts[k].contactId;
            writeRow(world, contactId);
            const so = c * SLOT_STRIDE;
            slot[so] = contactId;
            slot[so + 1] = gm;
            slot[so + 2] = gp;
            contactExtent(world, contactId);
            gm += extent.manifolds;
            gp += extent.points;
            ++c;
        }
    }

    // Overflow region.
    const overflow = world.constraintGraph.colors[OVERFLOW_INDEX].contacts;
    for (let k = 0; k < overflow.length; ++k) {
        const contactId = overflow[k].contactId;
        writeRow(world, contactId);
        const so = c * SLOT_STRIDE;
        slot[so] = contactId;
        slot[so + 1] = gm;
        slot[so + 2] = gp;
        contactExtent(world, contactId);
        gm += extent.manifolds;
        gp += extent.points;
        ++c;
    }
}

/**
 * Collect the contacts the kernel `store` flagged for a hit event this step (the flag lives in each
 * contact's directory record). The solved impulses are already back in the persistent pool manifolds
 * — the kernel `store` wrote them there directly — so nothing else reads back. Walks every touching
 * contact; most scenes flag none.
 */
export function readbackHitEvents(
    world: WorldState,
    layout: SolveLayout,
    context: StepContext,
): void {
    const store = world.manifoldStore;
    const set = context.hitEventContacts;
    for (const span of layout.colors) {
        for (const contactId of span.color.convexContacts) {
            if (store.hit(contactId)) set.add(contactId);
        }
        for (const spec of span.color.contacts) {
            if (store.hit(spec.contactId)) set.add(spec.contactId);
        }
    }
    for (const spec of world.constraintGraph.colors[OVERFLOW_INDEX].contacts) {
        if (store.hit(spec.contactId)) set.add(spec.contactId);
    }
}
