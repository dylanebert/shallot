import { contactHit } from "../collision/manifoldstore";
import { OVERFLOW_INDEX } from "../common/constants";
import type { BodySimRef } from "../kernel/bodycolumns";
import { COLOR_SPAN_STRIDE, type Columns } from "../kernel/columns";
import { kernel } from "../kernel/kernel";
import type { WorldState } from "../world/world";
import { graphContacts } from "./graph";
import type { Softness } from "./softness";

/** The per-step solver context threaded through the solve (b3StepContext, scalar subset). */
export type StepContext = {
    world: WorldState;
    sims: BodySimRef[];
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
    awakeIslands: boolean[];
    splitIslandId: number;
    splitSleepTime: number;
    bulletBodies: BodySimRef[];
    hitEventContacts: Set<number>;
    jointEventFlags: Set<number>;
};

/** One active graph color's transient ranges; `colorIndex` selects its kernel joint array. */
export type ColorSpan = {
    colorIndex: number;
    wideStart: number;
    wideCount: number;
    meshStart: number;
    meshCount: number;
};

/** Counts to reserve the solver columns, and ranges for scheduling the colored solve. */
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
const layoutViews = new WeakMap<WorldState, Uint32Array>();

/** Derive reservation sizes and scheduling ranges from the kernel graph. The pooled result is
 * consumed synchronously within this step, before another world computes its layout. */
export function computeLayout(world: WorldState): SolveLayout {
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    const ptr = k.graphComputeLayout();
    let words = layoutViews.get(world);
    if (words === undefined || words.buffer !== k.memory.buffer || words.byteOffset !== ptr) {
        words = new Uint32Array(k.memory.buffer, ptr, 10 + OVERFLOW_INDEX * 5);
        layoutViews.set(world, words);
    }
    const layout = layoutScratch;
    layout.contacts = words[0];
    layout.manifolds = words[1];
    layout.points = words[2];
    layout.wide = words[3];
    layout.meshStart = words[4];
    layout.meshTotal = words[5];
    layout.wideTotal = words[6];
    layout.overflowStart = words[7];
    layout.overflowCount = words[8];
    const count = words[9];
    for (let i = 0; i < count; ++i) {
        const o = 10 + i * 5;
        let span = spansScratch[i];
        if (span === undefined) {
            span = { colorIndex: 0, wideStart: 0, wideCount: 0, meshStart: 0, meshCount: 0 };
            spansScratch[i] = span;
        }
        span.colorIndex = words[o];
        span.wideStart = words[o + 1];
        span.wideCount = words[o + 2];
        span.meshStart = words[o + 3];
        span.meshCount = words[o + 4];
    }
    spansScratch.length = count;
    return layout;
}

/** Publish ranges to the kernel's batched jointless color loop. */
export function writeColorSpans(cols: Columns, layout: SolveLayout): void {
    const span = cols.colorSpan;
    for (let i = 0; i < layout.colors.length; ++i) {
        const c = layout.colors[i];
        const o = i * COLOR_SPAN_STRIDE;
        span[o] = c.wideStart;
        span[o + 1] = c.wideCount;
        span[o + 2] = c.meshStart;
        span[o + 3] = c.meshCount;
        span[o + 4] = 0;
        span[o + 5] = 0;
    }
}

/** Gather the graph's contact ids into transient solver slots, ordered convex, mesh, overflow. */
export function writeSlots(world: WorldState): void {
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    k.graphWriteSlots();
}

/** Collect contact hits after the kernel stores impulses into their persistent manifolds. */
export function readbackHitEvents(
    world: WorldState,
    layout: SolveLayout,
    context: StepContext,
): void {
    const dirU = world.manifoldStore.dirU;
    const set = context.hitEventContacts;
    for (const span of layout.colors) {
        const convex = graphContacts(world, span.colorIndex);
        for (let j = 0; j < convex.length; ++j) {
            if (contactHit(dirU, convex[j])) set.add(convex[j]);
        }
        const contacts = graphContacts(world, span.colorIndex, true);
        for (let j = 0; j < contacts.length; j += 2) {
            if (contactHit(dirU, contacts[j])) set.add(contacts[j]);
        }
    }
    const overflow = graphContacts(world, OVERFLOW_INDEX, true);
    for (let j = 0; j < overflow.length; j += 2) {
        if (contactHit(dirU, overflow[j])) set.add(overflow[j]);
    }
}
