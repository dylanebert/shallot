import type { Kernel } from "../kernel/kernel";

/** Box3D's b3Counters. Graph color counts include contacts and joints. */
export interface PhysicsWorldCounters {
    bodyCount: number;
    shapeCount: number;
    contactCount: number;
    jointCount: number;
    islandCount: number;
    stackUsed: number;
    arenaCapacity: number;
    staticTreeHeight: number;
    treeHeight: number;
    satCallCount: number;
    satCacheHitCount: number;
    /** Live kernel-allocator bytes in this World's kernel instance, excluding JavaScript storage.
     * Box3D instead reports allocator bytes across its entire process. */
    byteCount: number;
    taskCount: number;
    colorCounts: number[];
    manifoldCounts: number[];
    awakeContactCount: number;
    recycledContactCount: number;
    distanceIterations: number;
    pushBackIterations: number;
    rootIterations: number;
}

export class WorldCountersBuffer {
    private readonly _result: PhysicsWorldCounters = {
        bodyCount: 0,
        shapeCount: 0,
        contactCount: 0,
        jointCount: 0,
        islandCount: 0,
        stackUsed: 0,
        arenaCapacity: 0,
        staticTreeHeight: 0,
        treeHeight: 0,
        satCallCount: 0,
        satCacheHitCount: 0,
        byteCount: 0,
        taskCount: 0,
        colorCounts: Array<number>(24).fill(0),
        manifoldCounts: Array<number>(8).fill(0),
        awakeContactCount: 0,
        recycledContactCount: 0,
        distanceIterations: 0,
        pushBackIterations: 0,
        rootIterations: 0,
    };
    private _words = new Int32Array(0);

    read(k: Kernel, world: number): PhysicsWorldCounters {
        const pointer = k.worldCounters(world);
        if (this._words.buffer !== k.memory.buffer || this._words.byteOffset !== pointer)
            this._words = new Int32Array(k.memory.buffer, pointer, 50);
        const words = this._words;
        const result = this._result;
        result.bodyCount = words[0]!;
        result.shapeCount = words[1]!;
        result.contactCount = words[2]!;
        result.jointCount = words[3]!;
        result.islandCount = words[4]!;
        result.stackUsed = words[5]!;
        result.arenaCapacity = words[6]!;
        result.staticTreeHeight = words[7]!;
        result.treeHeight = words[8]!;
        result.satCallCount = words[9]!;
        result.satCacheHitCount = words[10]!;
        result.byteCount = words[11]!;
        result.taskCount = words[12]!;
        for (let i = 0; i < 24; i++) result.colorCounts[i] = words[13 + i]!;
        for (let i = 0; i < 8; i++) result.manifoldCounts[i] = words[37 + i]!;
        result.awakeContactCount = words[45]!;
        result.recycledContactCount = words[46]!;
        result.distanceIterations = words[47]!;
        result.pushBackIterations = words[48]!;
        result.rootIterations = words[49]!;
        return result;
    }
}
