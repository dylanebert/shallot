// Loader for the wasm-simd128 physics kernel (kernel/, inlined by scripts/build-kernel.ts).
//
// The kernel is ~tens of KB, too large for a synchronous main-thread compile, so instantiation is
// async: call `init(world)` once before the first `step()`. `step()` itself stays synchronous. Each owning
// engine World owns one kernel instance, memory and worker pool; the SoA columns and their TypeScript
// views therefore cannot overlap another StandardPhysicsPlugin world's memory. Standalone solver calls passing
// undefined retain one process-local kernel for the low-level World API.
//
// Two artifacts (scripts/build-kernel.ts). `init(world)` resolves threading itself: standalone (bun/node) and a
// cross-origin-isolated browser get the multithreaded artifact, which needs a shared `WebAssembly.Memory`;
// a browser without that isolation runs single-thread after one plain log naming the COOP/COEP headers the
// host is missing. `init(world, { threads })` is the advanced escape — 0 forces single-thread, n overrides the
// auto count. The MT artifact loads behind a dynamic `import()`, so a single-thread consumer never parses it.

import type { World } from "../../../engine";
import { KERNEL_WASM_BASE64 } from "./kernel.wasm";
import { createPool, maxWorkers, type Pool } from "./pool";

/** The kernel's exported surface — grows as each solver phase ports to wasm. */
export type Kernel = {
    memory: WebAssembly.Memory;
    viewEpochPtr(): number;
    activeWorld(): number;
    /** Toolchain smoke buffer offset + scale, the standing wasm-simd128 cliff gate (kernel.test.ts). */
    scratchPtr(): number;
    /** Per-shape scratch input: transform(7), count/radius, translation(3), fraction/encroach,
     * then up to 128 xyz points. Cast output: hit, fraction, point(3), normal(3), iterations,
     * triangle, child, material. Mover output is caller-owned: normal(3), offset, point(3),
     * then three i32 indices, 40 bytes per plane. `local` selects per-kind dispatch without a
     * shape transform, preserving the local oracle's signed zeros. */
    shapeQueryInputPtr(): number;
    shapeQueryOutputPtr(): number;
    shapeQueryRay(world: number, shape: number, local: number): void;
    shapeQueryCast(world: number, shape: number, local: number): void;
    shapeQueryOverlap(world: number, shape: number): number;
    shapeQueryMover(
        world: number,
        shape: number,
        output: number,
        capacity: number,
        local: number,
    ): number;
    worldQueryHeaderPtr(): number;
    worldQueryResultPtr(): number;
    worldQuery(world: number, operation: number, callback: number): void;
    bodyQuery(world: number, operation: number, head: number, capacity: number): void;
    sensorQuery(world: number, sensor: number): number;
    smokeScale(len: number, k: number): void;

    // Shared-column arena (kernel/src/arena.rs). `reserve` lays out the columns for one step's counts
    // and may grow memory; `layoutPtr` returns the byte-offset header the TS views derive from (see
    // columns.ts). The phase shims drive the solve over those columns, one phase at a time.
    reserve(
        body: number,
        contact: number,
        manifold: number,
        point: number,
        wide: number,
        color: number,
    ): void;
    layoutPtr(): number;

    // Allocator-owned body columns for the selected World, sized to its body high-water.
    reserveBodies(cap: number): number;
    solverSetTransferBody(
        source: number,
        index: number,
        target: number,
        flags: number,
        head: number,
        clearTransient: number,
    ): number;
    solverSetWakeBody(source: number, index: number, flags: number, head: number): number;
    solverSetRemoveBody(source: number, index: number): number;
    solverSetCopyBody(source: number, index: number, target: number, destination: number): void;
    solverSetMoveContact(source: number, index: number, target: number): number;
    solverSetSleepContact(id: number, target: number): void;
    solverSetMoveIsland(source: number, index: number, target: number): number;
    islandSplit(id: number): void;
    islandCreate(set: number): number;
    islandDestroy(id: number): void;
    islandCount(): number;
    islandField(id: number, field: number): number;
    islandSetField(id: number, field: number, value: number): void;
    islandArrayCount(id: number, kind: number): number;
    islandArrayGet(id: number, kind: number, index: number, lane: number): number;
    islandAddBody(id: number, body: number): void;
    islandRemoveBody(id: number, index: number): void;
    islandLinkContact(id: number, a: number, b: number): void;
    islandUnlinkContact(id: number): void;
    solverSetCreate(): number;
    solverSetCount(): number;
    solverSetIndex(id: number): number;
    solverSetDestroy(id: number): void;
    solverSetBodyCount(id: number): number;
    solverSetBodyId(set: number, index: number): number;
    solverSetBodyAppend(id: number): number;
    solverSetBodyPop(id: number): void;
    solverSetLayout(id: number): number;
    solverSetArrayCount(id: number, kind: number): number;
    solverSetArrayGet(id: number, kind: number, index: number): number;
    solverSetArrayPush(id: number, kind: number, value: number): number;
    solverSetArrayRemove(id: number, kind: number, index: number): number;
    solverSetArrayWrite(id: number, kind: number, index: number, value: number): void;
    solverSetArrayPop(id: number, kind: number): void;
    bodyLayoutPtr(): number;
    bodySetEntity(world: number, body: number, eid: number): void;
    bodySyncMoved(count: number): number;
    /** The record capacity the resident body region is sized to — the single source of truth for the
     * TS body-store's column-view lengths (bodycolumns.ts). Zero before the first `reserveBodies`. */
    bodyCap(): number;
    /** Set the world context used by kernel finalization's move publication. */
    bodySetActiveWorld(world: number): void;
    /** Select the World's persistent layouts for the next kernel operation; no bytes are copied. */
    shapeSetActiveWorld(world: number): void;
    /** Allocate a body index/generation from the kernel-owned world-local pool. */
    bodyCreate(world: number): number;
    bodyCreateSim(
        world: number,
        type: number,
        flags: number,
        awake: boolean,
        enabled: boolean,
        threshold: number,
        px: number,
        py: number,
        pz: number,
        qx: number,
        qy: number,
        qz: number,
        qs: number,
        vx: number,
        vy: number,
        vz: number,
        wx: number,
        wy: number,
        wz: number,
        linearDamping: number,
        angularDamping: number,
        gravityScale: number,
    ): number;
    /** Release a body index into the kernel-owned world-local free list. */
    bodyDestroy(world: number, id: number): number;
    /** Clear a world-local body pool after the public world is destroyed. */
    bodyResetWorld(world: number): void;
    residentResetWorld(world: number): void;
    worldSnapshot(world: number): number;
    worldSnapshotBuffer(bytes: number): number;
    worldRestore(world: number): void;
    bodyGeneration(world: number, id: number): number;
    bodyAlive(world: number, id: number): number;
    bodyCount(world: number): number;
    bodyLength(world: number): number;
    bodyFinish(count: number, timeStep: number, enableSleep: boolean): number;
    bodyVelocitySet(
        world: number,
        id: number,
        angular: boolean,
        x: number,
        y: number,
        z: number,
    ): boolean;
    bodyApply(
        world: number,
        id: number,
        kind: number,
        x: number,
        y: number,
        z: number,
        px: number,
        py: number,
        pz: number,
        maxSpeed: number,
    ): void;
    bodySetPose(
        world: number,
        id: number,
        x: number,
        y: number,
        z: number,
        qx: number,
        qy: number,
        qz: number,
        qs: number,
    ): void;
    shapeQueryPose(world: number, shape: number, body: number): void;
    shapeSyncBodyBounds(world: number, body: number): void;
    shapeBodyAllowsType(world: number, body: number, type: number): number;
    shapeBodyTake(world: number, body: number): number;
    bodyTransfer(world: number, id: number, target: number, clearTransient: boolean): number;
    bodyWakeRecord(world: number, id: number): void;
    bodyCreateContact(
        world: number,
        shapeA: number,
        shapeB: number,
        child: number,
        flags: number,
    ): number;
    bodyDestroyContact(world: number, id: number): void;
    bodySyncFlags(world: number, id: number): void;
    bodyChangeType(world: number, id: number, type: number): void;
    bodyUpdateMass(world: number, id: number): void;
    bodyCreateIsland(world: number, id: number): void;
    bodyRemoveIsland(world: number, id: number): void;
    bodyColumnPtr(world: number, id: number, column: number): number;
    simColumnPtr(world: number, set: number, index: number, column: number): number;
    bodyStateIndex(world: number, id: number): number;
    bodySyncContacts(world: number, id: number): void;
    bodyTargetVelocity(
        world: number,
        id: number,
        tx: number,
        ty: number,
        tz: number,
        qx: number,
        qy: number,
        qz: number,
        qs: number,
        timeStep: number,
        wake: boolean,
    ): boolean;
    islandCanSleep(id: number): boolean;
    islandSplitCandidate(): number;
    islandSetSplitCandidate(id: number): void;

    // One allocator-owned fat AABB per shape in the selected World.
    reserveFatAabb(cap: number): number;
    fatAabbLayoutPtr(): number;
    fatAabbCap(): number;

    // Allocator-owned shape records and lifecycle columns for the selected World.
    reserveShapes(cap: number): number;
    shapeLayoutPtr(): number;
    shapeCap(): number;
    /** Allocate/release a world-local shape slot; generation and validity stay in wasm. */
    shapeCreate(
        world: number,
        body: number,
        type: number,
        density: number,
        explosion: number,
        flags: number,
    ): number;
    shapeCanCreate(world: number, body: number, type: number): boolean;
    shapeSetGeometry(
        world: number,
        id: number,
        a: number,
        b: number,
        c: number,
        d: number,
        e: number,
        f: number,
        g: number,
    ): number;
    shapeGeometryOutputPtr(): number;
    shapeMaterialIndex(world: number, shape: number, child: number, triangle: number): number;
    shapeCompoundChild(world: number, shape: number, child: number): number;
    shapeCompoundChildType(world: number, shape: number, child: number): number;
    shapeComputeMass(world: number, id: number): void;
    shapeComputeExtent(world: number, id: number, x: number, y: number, z: number): void;
    shapeGetCentroid(world: number, id: number): void;
    computeQuatBetween(
        ax: number,
        ay: number,
        az: number,
        bx: number,
        by: number,
        bz: number,
    ): number;
    shapeFindHullSupportVertex(world: number, id: number, x: number, y: number, z: number): number;
    shapeFindHullSupportFace(world: number, id: number, x: number, y: number, z: number): number;
    shapeQueryCompound(
        world: number,
        id: number,
        lx: number,
        ly: number,
        lz: number,
        ux: number,
        uy: number,
        uz: number,
    ): void;
    shapeComputeAABB(
        world: number,
        id: number,
        x: number,
        y: number,
        z: number,
        qx: number,
        qy: number,
        qz: number,
        qs: number,
        extra: number,
    ): void;
    shapeFinishGeometry(world: number, id: number): void;
    shapeFilterWrite(
        world: number,
        id: number,
        categoryHi: number,
        categoryLo: number,
        maskHi: number,
        maskLo: number,
        group: number,
    ): void;
    shapeContactNext(world: number, id: number, key: number): number;
    shapeAttachSensor(world: number, id: number, sensor: number): void;
    shapeLink(world: number, id: number, body: number): void;
    shapeUnlink(world: number, id: number): void;
    shapeSyncBody(world: number, body: number): void;
    shapeCreateProxy(world: number, id: number, force: number): void;
    shapeCreateProxyTransform(
        world: number,
        id: number,
        force: number,
        x: number,
        y: number,
        z: number,
        qx: number,
        qy: number,
        qz: number,
        qs: number,
    ): void;
    shapeDestroyProxy(world: number, id: number): void;
    shapeBodyProxies(world: number, body: number, create: number): void;
    shapeDestroy(world: number, id: number): void;
    shapeResetWorld(world: number): void;
    shapeGeneration(world: number, id: number): number;
    shapeAlive(world: number, id: number): number;
    shapeCount(world: number): number;
    /** Kernel-owned live material records attached to shape slots. */
    shapeAllocateMaterials(world: number, id: number, count: number): number;
    shapeFreeMaterials(world: number, id: number): void;
    shapeMaterialPtr(world: number, id: number): number;
    shapeMaterialSet(
        world: number,
        id: number,
        index: number,
        friction: number,
        restitution: number,
        rolling: number,
        x: number,
        y: number,
        z: number,
        low: number,
        high: number,
        color: number,
    ): void;
    shapeMaterialCount(world: number, id: number): number;

    // World-local tree pools, pair membership and moves. Pass zero to retain a capacity.
    reserveBroad(capS: number, capK: number, capD: number, setCap: number): number;
    bodyShouldBodiesCollide(bodyA: number, bodyB: number): number;
    broadLayoutPtr(): number;
    broadTreeCap(i: number): number;
    reserveTreeWork(depth: number, words: number): number;
    treeEnlargePass(count: number, bullets: number): void;
    treeCreateProxy(
        index: number,
        lx: number,
        ly: number,
        lz: number,
        hx: number,
        hy: number,
        hz: number,
        hi: number,
        lo: number,
        user: number,
    ): number;
    treeDestroyProxy(index: number, id: number): void;
    treeEnlargeProxy(
        index: number,
        id: number,
        lx: number,
        ly: number,
        lz: number,
        hx: number,
        hy: number,
        hz: number,
    ): void;
    treeMoveProxy(
        index: number,
        id: number,
        lx: number,
        ly: number,
        lz: number,
        hx: number,
        hy: number,
        hz: number,
    ): void;
    broadBufferMove(key: number): void;
    broadClearMoves(): void;
    treeMutateResident(
        index: number,
        op: number,
        id: number,
        lx: number,
        ly: number,
        lz: number,
        hx: number,
        hy: number,
        hz: number,
        ch: number,
        cl: number,
        ud: number,
        udh: number,
    ): number;
    treeMutate(
        ptr: number,
        cap: number,
        state: number,
        op: number,
        id: number,
        lx: number,
        ly: number,
        lz: number,
        hx: number,
        hy: number,
        hz: number,
        ch: number,
        cl: number,
        ud: number,
        udh: number,
    ): number;
    treeQuery(
        ptr: number,
        cap: number,
        root: number,
        count: number,
        lx: number,
        ly: number,
        lz: number,
        hx: number,
        hy: number,
        hz: number,
        mh: number,
        ml: number,
        all: number,
        state: number,
    ): void;
    broadCreateProxy(
        type: number,
        lx: number,
        ly: number,
        lz: number,
        hx: number,
        hy: number,
        hz: number,
        categoryHi: number,
        categoryLo: number,
        shape: number,
        force: number,
    ): number;
    broadTestOverlap(a: number, b: number): number;
    broadDestroyProxy(key: number): void;
    broadMoveProxy(
        key: number,
        lx: number,
        ly: number,
        lz: number,
        hx: number,
        hy: number,
        hz: number,
    ): void;
    broadEnlargeProxy(
        key: number,
        lx: number,
        ly: number,
        lz: number,
        hx: number,
        hy: number,
        hz: number,
    ): void;
    broadClearMoved(type: number, id: number): void;
    broadCreateSet(capacity: number): void;
    broadSetCap(): number;
    broadAddPair(a: number, b: number, child: number): number;
    broadRemovePair(a: number, b: number, child: number): number;

    // `reservePairs` reserves move-result lists and rebuild scratch. ParKind.Pairs queries the
    // resident proxies, shapes, body types and pair table, including compound recursion and default
    // filtering. `pairsOverflow` grows survivor capacity for a read-only retry after overflow.
    // Each result is (child, shapeA, shapeB, next), prepended at its proxy's `pairsCandEndPtr` head
    // (u32::MAX for empty). TypeScript creates contacts from those lists in place after the join.
    reservePairs(): void;
    pairsCandEndPtr(): number;
    pairsCandPtr(): number;
    pairsOverflow(): number;
    rebuildTrees(): void;

    // Immutable native geometry images are retained by caller identity inside each world; shapes
    // and compounds hold the resulting kernel address. Hulls remain content-interned separately.
    hullUploadBuffer(world: number, bytes: number): number;
    hullDatabaseAdd(world: number, bytes: number): number;
    hullDatabaseLookup(world: number, bytes: number): number;
    hullDatabaseRemove(world: number, handle: number): void;
    hullDatabaseCount(world: number): number;
    hullDatabaseRefs(world: number, handle: number): number;
    geometryUploadBuffer(world: number, bytes: number): number;
    geometryDatabaseAdd(
        world: number,
        kind: number,
        identity: number,
        bytes: number,
        refs: number,
    ): number;
    geometryDatabaseLookup(world: number, kind: number, identity: number): number;
    geometryDatabaseRemove(world: number, kind: number, pointer: number): void;
    geometryDatabaseIdentity(world: number, kind: number, pointer: number): number;
    geometryDatabaseCount(world: number): number;
    geometryDatabaseAllocationBytes(world: number): number;
    geometryDatabaseRefs(world: number, kind: number, pointer: number): number;

    // The contact-id directory grows on allocation. Its block addresses remain stable while the
    // manifold-count allocators grow; this layout header exposes only the directory's address.
    contactRecordCapacity(worldId: number): number;
    manifoldAllocatorOperations(worldId: number): bigint;
    allocateManifolds(contactId: number, count: number): number;
    freeManifolds(contactId: number): void;
    manifoldLayoutPtr(): number;
    collideHullsGeo(
        a: number,
        b: number,
        px: number,
        py: number,
        pz: number,
        qx: number,
        qy: number,
        qz: number,
        qs: number,
    ): number;
    geoOutPtr(): number;
    geoTriangleOutPtr(): number;
    collideSpheresGeo(
        ax: number,
        ay: number,
        az: number,
        ar: number,
        bx: number,
        by: number,
        bz: number,
        br: number,
        px: number,
        py: number,
        pz: number,
        qx: number,
        qy: number,
        qz: number,
        qs: number,
    ): number;

    reserveCollide(count: number, threads: number, defaultMix: number, distance: number): void;
    collideListPtr(): number;
    contactStatePtr(): number;
    contactPairOrder(typeA: number, typeB: number): number;
    allocContact(): number;
    freeContact(contactId: number): void;
    contactCapacity(worldId: number): number;
    contactCount(worldId: number): number;
    graphComputeLayout(): number;
    graphWriteSlots(): void;
    graphCreate(capacity: number): void;
    graphBodyBit(color: number, id: number): number;
    graphAssignColor(a: number, b: number, ta: number, tb: number): number;
    graphClearBodies(color: number, a: number, b: number): void;
    graphContactCount(color: number, scalar: number): number;
    graphContactPtr(color: number, scalar: number): number;
    graphAddContact(id: number, indexA: number, indexB: number): void;
    graphRemoveContact(a: number, b: number, color: number, index: number, mesh: number): void;
    jointCreate(
        a: number,
        b: number,
        type: number,
        drawScale: number,
        collideConnected: number,
        ax: number,
        ay: number,
        az: number,
        aqx: number,
        aqy: number,
        aqz: number,
        aqs: number,
        bx: number,
        by: number,
        bz: number,
        bqx: number,
        bqy: number,
        bqz: number,
        bqs: number,
        force: number,
        torque: number,
        hertz: number,
        damping: number,
    ): number;
    jointDestroy(id: number, wake: number): void;
    jointTransfer(id: number, target: number): void;
    jointLink(id: number): void;
    jointUnlink(id: number): void;
    solverSetWake(set: number): void;
    jointCollectEvents(): number;
    jointEventPtr(): number;
    awakeContactCount(): number;
    awakeContactGet(index: number): number;
    awakeContactCopy(ptr: number): void;
    awakeContactUpdate(id: number): void;
    awakeContactRemove(id: number): void;
    jointRecordCount(): number;
    jointRecordCapacity(): number;
    jointRecordPtr(): number;
    jointSimPtr(id: number): number;
    jointArrayCount(key: number): number;
    jointArrayPtr(key: number): number;
    jointReadWord(key: number, index: number, field: number): number;
    jointWriteWord(key: number, index: number, field: number, value: number): void;
    meshCacheCapacity(worldId: number): number;
    ensureMeshCache(contactId: number): void;
    freeMeshCache(contactId: number): void;
    dispatchContacts(count: number): void;
    continuousPtr(): number;
    continuousRoots(s: number, k: number, d: number, enableSleep: boolean): void;

    // Solve columns are reserved while workers are parked. With no pool,
    // threadCount is one and runMt executes every stage inline, including pose finalization.
    solveBuild(
        threadCount: number,
        subStepCount: number,
        wideTotal: number,
        meshStart: number,
        meshTotal: number,
        overflowStart: number,
        overflowCount: number,
        jointTotal: number,
        overflowJointCount: number,
        gx: number,
        gy: number,
        gz: number,
        h: number,
        invH: number,
        dt: number,
        invDt: number,
        maxLinearVelocity: number,
        contactSpeed: number,
        csBias: number,
        csMass: number,
        csImpulse: number,
        ssBias: number,
        ssMass: number,
        ssImpulse: number,
        warmStartScale: number,
        restitutionThreshold: number,
        hitEventThreshold: number,
        enableContinuous: number,
    ): void;
    // Flat block-claim sweeps share the pool. The kernel prices the fork floor: 1 forks, 0 runs the
    // same task on the caller. Build immediately precedes run; `a` carries a phase parameter.
    parBuild(kind: ParKind, count: number, threadCount: number, a: number): number;
    /** Run the built job (staged solve or parallel-for) on the calling thread — the orchestrator. */
    runMt(): void;
    /** Run the built job as pooled worker `index` (1-based). Exactly once per round. */
    workerMain(index: number): void;
    /** Abandon the running solve — a worker trapped. Breaks the orchestrator's wasm-side spins, which
     * no JS event can reach; the worker's round body calls it before it acks (pool.ts). */
    workerFault(): void;
};

/** Which outer phase a {@link KernelExports.parBuild} names (kernel/src/solve.rs `Job`). */
export const ParKind = {
    Contacts: 2,
    Bullets: 3,
    Pairs: 4,
} as const;
export type ParKind = (typeof ParKind)[keyof typeof ParKind];

/** Options for {@link init}. */
export type InitOptions = {
    /**
     * the advanced escape from the default threading. {@link init} multithreads on its own wherever the
     * host allows it, so this is rarely needed: pass `0` to force the single-thread kernel, or `n` to
     * override the auto count (counting the calling thread, clamped to the ceiling the shadow stack
     * affords). `n` still needs a host that can hold shared memory — a browser without cross-origin
     * isolation runs single-thread whatever you ask. read {@link threads} for what you got. the
     * multithreaded kernel caps linear memory at 1 GiB (the single-thread one is unbounded), which bounds
     * the scene it can hold.
     */
    threads?: number;
};

export type QueryCallback = (kind: number, shape: number, data: number, count: number) => number;

export interface KernelState {
    queryCallback: QueryCallback | null;
    queryWorld: number;
    callbackDepth: number;
    queryFailed: boolean;
    queryError: unknown;
    instance: Kernel | null;
    sharedMemory: WebAssembly.Memory | null;
    pool: Pool | null;
    resolved: number;
    booting: Promise<void> | null;
    dead: boolean;
    viewEpoch: Uint32Array | null;
    viewValue: number;
    viewRevision: number;
}

function createKernelState(): KernelState {
    return {
        queryCallback: null,
        queryWorld: -1,
        callbackDepth: 0,
        queryFailed: false,
        queryError: undefined,
        instance: null,
        sharedMemory: null,
        pool: null,
        resolved: 1,
        booting: null,
        dead: false,
        viewEpoch: null,
        viewValue: -1,
        viewRevision: 0,
    };
}

const kernelStateKey = { create: createKernelState };
const standaloneKernelState = createKernelState();
const queryImportStates = new WeakMap<Kernel, KernelState>();

export function kernelState(world: World | undefined): KernelState {
    return world ? world.resource(kernelStateKey) : standaloneKernelState;
}

/** Whether no view over `world`'s kernel memory changed since `state.viewRevision` was taken. */
export function kernelViewsCurrent(world: World | undefined, state: KernelState): boolean {
    const epoch = state.viewEpoch;
    // Shared memory compares the cached buffer, as Emscripten's growMemViews does before each heap
    // access. Unshared growth detaches the cached buffer, whose epoch view then reads undefined.
    return (
        epoch !== null &&
        !state.dead &&
        world?.disposed !== true &&
        (state.sharedMemory === null || epoch.buffer === state.sharedMemory.buffer) &&
        epoch[0] === state.viewValue
    );
}

/** One staleness key for every store on this kernel, including reallocations within existing pages. */
export function kernelViewKey(world: World | undefined, state = kernelState(world)): number {
    if (kernelViewsCurrent(world, state)) return state.viewRevision;
    const k = kernel(world);
    const buffer = k.memory.buffer;
    if (state.viewEpoch === null || state.viewEpoch.buffer !== buffer) {
        state.viewEpoch = new Uint32Array(buffer, k.viewEpochPtr(), 1);
        state.viewRevision++;
    }
    const value = state.viewEpoch[0];
    if (value !== state.viewValue) {
        state.viewValue = value;
        state.viewRevision++;
    }
    return state.viewRevision;
}

export function setQueryCallback(
    world: World | undefined,
    callback: QueryCallback | null,
): QueryCallback | null {
    const state = kernelState(world);
    const previous = state.queryCallback;
    state.queryCallback = callback;
    return previous;
}

/** @internal Snapshot of the guard and the state bound to this kernel's imported query callback. */
export function queryCallbackState(world: World | undefined) {
    const state = kernelState(world);
    const imported = queryImportStates.get(kernel(world));
    return {
        callbackDepth: state.callbackDepth,
        queryWorld: state.queryWorld,
        importMatchesGuard: imported === state,
        importCallbackDepth: imported?.callbackDepth,
        importQueryWorld: imported?.queryWorld,
    };
}

export function assertQueryWorld(world: World | undefined, worldId: number): void {
    const state = kernelState(world);
    if (state.callbackDepth !== 0 && state.queryWorld !== worldId)
        throw new Error("physics: one kernel cannot interleave two worlds' queries");
    state.queryWorld = worldId;
}

export function rethrowQueryError(world: World | undefined): void {
    const state = kernelState(world);
    if (!state.queryFailed) return;
    const error = state.queryError;
    state.queryFailed = false;
    state.queryError = undefined;
    throw error;
}

function queryImport(runtime: KernelState): QueryCallback {
    return (kind, shape, data, count) => {
        if (runtime.queryFailed) return 0;
        ++runtime.callbackDepth;
        try {
            if (runtime.queryCallback === null)
                throw new Error("physics: query callback is not installed");
            return runtime.queryCallback(kind, shape, data, count);
        } catch (error) {
            // Exceptions must not abandon Rust frames and leak the WASM shadow stack.
            runtime.queryFailed = true;
            runtime.queryError = error;
            return 0;
        } finally {
            --runtime.callbackDepth;
        }
    };
}

function decode(base64: string): Uint8Array<ArrayBuffer> {
    const bin = atob(base64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; ++i) bytes[i] = bin.charCodeAt(i);
    return bytes;
}

/** Threads {@link init} runs by default when the host allows it. Flat, not
 * scale-aware: 4 is optimal or within noise at every scene size and never regresses a small one, where
 * more threads would — the wake cost outweighs the split. */
export const AUTO_THREADS = 4;

/** The one-time log a browser gets when it blocks multithreading for want of cross-origin isolation.
 * Loud and host-actionable — the fix is the host's headers, not the caller's code. */
export const COOP_COEP_HINT =
    "physics: running single-threaded. Multithreading needs a cross-origin-isolated page — serve it with COOP and COEP headers (Cross-Origin-Opener-Policy: same-origin, Cross-Origin-Embedder-Policy: require-corp).";

/** The resolved threading plan: `want` threads to attempt (counting the caller; 0 is single-thread), and
 * whether a browser blocked multithreading (the one case {@link announce} logs). */
export type Threading = { want: number; warn: boolean };

/** The host signals {@link resolve} branches on. `shared` — can this host hold a shared
 * `WebAssembly.Memory`: cross-origin isolation in a browser, `SharedArrayBuffer` existing standalone.
 * `browser` picks whether a blocked host earns the COOP/COEP hint (headers only fix a page). */
export type Host = { browser: boolean; shared: boolean };

/**
 * Resolve the threading plan from the caller's request and the host — the pure decision table (the seam a
 * unit test fakes). Standalone always multithreads (`SharedArrayBuffer` is unconditional in bun/node); a
 * browser multithreads when cross-origin isolated and otherwise runs single-thread with a warning; an
 * explicit `0` forces single-thread with no warning; `n` overrides the auto count (the pool clamps it to
 * the link bound). The warning fires only when a browser wanted threads and couldn't have them.
 */
export function resolve(threads: number | undefined, host: Host): Threading {
    if (threads === 0) return { want: 0, warn: false };
    const want = threads ?? AUTO_THREADS;
    if (host.shared) return { want, warn: false };
    return { want: 0, warn: host.browser };
}

/** Emit the one-time COOP/COEP log for a browser that blocked threads. Split from {@link resolve} so the
 * decision table stays pure. */
export function announce(plan: Threading): void {
    if (plan.warn) console.log(COOP_COEP_HINT);
}

/** This host's threading signals. Standalone is detected the way the pool spawns its workers — a real
 * `process.versions.node` (node, bun, deno 2's node compat; pool.ts's spawn branch) — so the resolver and
 * the spawn path cannot disagree. Keying "browser" on `crossOriginIsolated` being a boolean would misroute
 * deno, which exposes that global too, into the COOP/COEP warning. */
function host(): Host {
    const p = (globalThis as { process?: { versions?: { node?: unknown } } }).process;
    if (p?.versions?.node != null) {
        return { browser: false, shared: typeof SharedArrayBuffer !== "undefined" };
    }
    const g = globalThis as { crossOriginIsolated?: boolean };
    return { browser: true, shared: g.crossOriginIsolated === true };
}

async function single(runtime: KernelState): Promise<void> {
    const result = await WebAssembly.instantiate(decode(KERNEL_WASM_BASE64), {
        env: { queryCallback: queryImport(runtime) },
    });
    const instance = result.instance.exports as unknown as Kernel;
    queryImportStates.set(instance, runtime);
    runtime.instance ??= instance;
}

async function multi(runtime: KernelState, want: number): Promise<void> {
    const { KERNEL_SHARED_WASM_BASE64, SHARED_INITIAL_PAGES, SHARED_MAX_PAGES, SHARED_STACK_SIZE } =
        await import("./kernel.shared.wasm");

    // The module declares its own memory floor (data + shadow stack), so `initial` may not go below it;
    // `maximum` is the link-time ceiling and a JS memory may only lower it.
    const memory = new WebAssembly.Memory({
        initial: SHARED_INITIAL_PAGES,
        maximum: SHARED_MAX_PAGES,
        shared: true,
    });
    const module = await WebAssembly.compile(decode(KERNEL_SHARED_WASM_BASE64));
    const exports = (
        await WebAssembly.instantiate(module, {
            env: { memory, queryCallback: queryImport(runtime) },
        })
    ).exports as unknown as Kernel & {
        // biome-ignore lint/style/useNamingConvention: LLD's global, exported under its own name.
        __stack_pointer: WebAssembly.Global;
    };
    // A lazy `kernel(world)` can have run during those awaits; it wins rather than swapping memory out from
    // under views already held by this world's solver.
    if (runtime.instance) return;

    const count = Math.min(want, 1 + maxWorkers(SHARED_STACK_SIZE)) - 1;
    // Instantiating ran the start function to completion on THIS thread, which is the ordering the pool
    // depends on: it CAS-guards data init, and a thread that loses that CAS blocks on
    // `memory.atomic.wait32` — which traps on a browser main thread. Workers may only spawn after it, and
    // they take the main instance's exports as their argument, so they cannot spawn before it exists.
    const spawned =
        count > 0
            ? await createPool(
                  module,
                  memory,
                  count,
                  exports.__stack_pointer.value as number,
                  SHARED_STACK_SIZE,
              )
            : null;

    runtime.instance = { ...exports, memory } as Kernel;
    queryImportStates.set(runtime.instance, runtime);
    runtime.sharedMemory = memory;
    runtime.pool = spawned;
    runtime.resolved = count + 1;
}

async function boot(runtime: KernelState, threads: number | undefined): Promise<void> {
    if (runtime.instance) return;
    const plan = resolve(threads, host());
    announce(plan);
    if (plan.want >= 1) {
        try {
            await multi(runtime, plan.want);
            return;
        } catch (e) {
            // A host with shared memory can still refuse the workers themselves (a CSP that blocks blob:
            // URLs, a worker-count limit). Fall back to single-thread rather than leaving the caller with
            // no physics at all — but never silently: an invisible perf cliff between deployments is what
            // the default-on design exists to avoid.
            console.log(
                `physics: running single-threaded. The host blocked the worker pool: ${e instanceof Error ? e.message : String(e)}`,
            );
            await runtime.pool?.terminate();
            runtime.pool = null;
            runtime.sharedMemory = null;
            runtime.instance = null;
            runtime.resolved = 1;
        }
    }
    await single(runtime);
}

/**
 * Instantiate the given World's physics kernel, or the standalone kernel for explicit `undefined`.
 * idempotent — subsequent calls resolve immediately, and the first call decides threading. await once before the first `step()`. required in a browser, where the main thread
 * refuses to compile a wasm module this size synchronously; outside a browser (bun/node/deno) `step()`
 * also instantiates lazily, so the await is optional there — but a lazy instance is single-threaded, so
 * `init(state)` before you touch a `World` if you want threads.
 *
 * threading resolves itself: standalone and a cross-origin-isolated browser multithread; a browser without
 * that isolation logs the missing COOP/COEP headers once and runs single-thread. Pass
 * {@link InitOptions.threads} only to force single-thread (`0`) or override the count. the worker pool
 * never holds the process open — a standalone script exits when its own work is done, no `shutdown(state)`
 * needed.
 *
 * @example
 * await init(state); // multithreaded wherever the host allows it
 * console.log(threads(state)); // what the host actually gave
 * @example
 * await init(state, { threads: 0 }); // force single-thread
 */
export function init(world: World | undefined, options?: InitOptions): Promise<void> {
    const runtime = kernelState(world);
    runtime.booting ??= boot(runtime, normalizeThreads(options?.threads));
    return runtime.booting;
}

/** Normalize the caller's `threads`: absent stays absent (auto-resolve), a non-finite value becomes 0
 * (single-thread), everything else floors to a non-negative integer. */
function normalizeThreads(v: number | undefined): number | undefined {
    if (v === undefined) return undefined;
    if (!Number.isFinite(v)) return 0;
    return Math.max(0, Math.floor(v));
}

/** threads the kernel resolved to — 1 when it is running single-threaded. */
export function threads(world: World | undefined): number {
    return kernelState(world).resolved;
}

/** Stop the worker pool; the kernel keeps stepping, single-threaded. Optional: the pooled workers are
 * `unref`'d at boot, so a script that inits, steps, and ends exits on its own without this (pool.ts). Call
 * it to release the worker threads deterministically — at a test suite's teardown, say. */
export async function shutdown(world: World | undefined): Promise<void> {
    const runtime = kernelState(world);
    await runtime.pool?.terminate();
    runtime.pool = null;
    runtime.resolved = 1;
}

/**
 * The kernel instance. Lazily instantiates synchronously if `init(state)` hasn't run — fine in bun/node/
 * deno; a browser main thread throws on a synchronous compile this large, so browser callers must
 * `await init(state)` first.
 */
export function kernel(world: World | undefined): Kernel {
    const runtime = kernelState(world);
    if (runtime.dead) {
        throw new Error(
            "physics kernel is dead: a worker trapped mid-step, so the shared columns hold a partial one",
        );
    }
    if (!runtime.instance) {
        if (runtime.booting) throw new Error("await init() before stepping");
        const mod = new WebAssembly.Module(decode(KERNEL_WASM_BASE64));
        runtime.instance = new WebAssembly.Instance(mod, {
            env: { queryCallback: queryImport(runtime) },
        }).exports as unknown as Kernel;
        queryImportStates.set(runtime.instance, runtime);
    }
    return runtime.instance;
}

/**
 * The shared memory's current byte length, or 0 single-threaded — the staleness key for views over the
 * kernel's columns.
 *
 * A `memory.grow` detaches every view over an unshared memory (`length === 0`, the single-thread
 * path's guard). A shared memory never detaches: grow hands back a *new* `SharedArrayBuffer` object
 * aliasing the same backing store, so an old view still reads and writes the correct physical bytes and
 * only misses the new tail. Views over the shared path therefore key staleness on this length changing.
 */
export function sharedBytes(world: World | undefined): number {
    const memory = kernelState(world).sharedMemory;
    return memory === null ? 0 : memory.buffer.byteLength;
}

/** The worker pool the solve may run on, or null when the kernel is single-threaded — or when a worker
 * has faulted, which kills the kernel (`runPool`). */
export function workers(world: World | undefined): Pool | null {
    const pool = kernelState(world).pool;
    return pool?.alive ? pool : null;
}

/**
 * Wake the pool for one round, and kill the kernel if a worker trapped inside it.
 *
 * A trapped worker stops claiming blocks *mid-sweep*, and every phase the pool drives writes state that
 * outlives the step: the staged solve's impulses, and — since the outer phases moved onto the pool — the
 * persistent manifold pool and contact directory. So the survivors are not the casualty; the columns are.
 * Retiring the pool and stepping on single-threaded would run the next step off a half-written manifold
 * store, which is silent corruption. The kernel is poisoned instead: the throw reaches the caller, and
 * every later call throws too.
 *
 * A trap here is a kernel bug (an out-of-bounds column access), not a condition a caller can handle —
 * there is nothing to recover to.
 */
export function runPool(
    world: World | undefined,
    pool: Pool,
    orchestrate: () => void,
    stableBlocks = false,
): void {
    try {
        pool.run(orchestrate, stableBlocks);
    } catch (e) {
        kernelState(world).dead = true;
        // The dead worker is gone; terminate the survivors to reclaim the threads (they are `unref`'d, so
        // they would not block exit, but they are live and now useless). Not awaited — this path is
        // already unwinding.
        void pool.terminate();
        throw e;
    }
}
