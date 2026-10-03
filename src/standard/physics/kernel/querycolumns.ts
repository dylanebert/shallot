import type { ShapeProxy } from "../collision/distance";
import { DEFAULT_CATEGORY_BITS, DEFAULT_MASK_BITS } from "../common/constants";
import type { AABB, Pos, Vec3, WorldTransform } from "../common/math";
import type { QueryFilter } from "../common/types";
import type { WorldState } from "../world/world";
import { rebuildGeometry } from "./geocolumns";
import { assertQueryWorld, type Kernel, kernel } from "./kernel";

export function queryColumns(world: WorldState): QueryColumns {
    return (world.queryColumns ??= new QueryColumns(world));
}

/** World-owned views and input upload for the kernel query ABI. */
export class QueryColumns {
    readonly world: WorldState;
    headerU = new Uint32Array(0);
    headerF = new Float32Array(0);
    input = new Float32Array(0);
    resultU = new Uint32Array(0);
    resultF = new Float32Array(0);
    cast = new Float32Array(0);

    constructor(world: WorldState) {
        this.world = world;
    }

    prepare(origin: Pos, filter?: QueryFilter): Kernel {
        const world = this.world;
        assertQueryWorld(world.ecsState, world.worldId);
        world.broadPhase.store.initialize();
        const k = kernel(world.ecsState);
        k.shapeSetActiveWorld(world.worldId);
        if (world.geometryDirty) {
            rebuildGeometry(world);
            world.geometryDirty = false;
        }
        const memory = k.memory.buffer;
        if (world.bodyStore.stale) world.bodyStore.refreshViews();
        if (world.shapeStore.shapeF.buffer !== memory) world.shapeStore.refreshViews();
        world.broadPhase.store.refreshIfStale();
        if (this.input.buffer !== memory || this.input.byteLength === 0) {
            this.input = new Float32Array(memory, k.shapeQueryInputPtr(), 398);
            this.cast = new Float32Array(memory, k.shapeQueryOutputPtr(), 12);
            this.headerU = new Uint32Array(memory, k.worldQueryHeaderPtr(), 19);
            this.headerF = new Float32Array(memory, k.worldQueryHeaderPtr(), 19);
            this.resultU = new Uint32Array(memory, k.worldQueryResultPtr(), 16);
            this.resultF = new Float32Array(memory, k.worldQueryResultPtr(), 16);
        }
        const h = this.headerU;
        const trees = world.broadPhase.trees;
        for (let i = 0; i < 3; ++i) {
            h[2 * i] = trees[i].root;
            h[2 * i + 1] = trees[i].nodeCount;
        }
        const category = filter?.categoryBits ?? DEFAULT_CATEGORY_BITS;
        const mask = filter?.maskBits ?? DEFAULT_MASK_BITS;
        h[6] =
            category === DEFAULT_CATEGORY_BITS
                ? 0xffffffff
                : Number((category >> 32n) & 0xffffffffn);
        h[7] = category === DEFAULT_CATEGORY_BITS ? 0xffffffff : Number(category & 0xffffffffn);
        h[8] = mask === DEFAULT_MASK_BITS ? 0xffffffff : Number((mask >> 32n) & 0xffffffffn);
        h[9] = mask === DEFAULT_MASK_BITS ? 0xffffffff : Number(mask & 0xffffffffn);
        this.headerF[10] = origin.x;
        this.headerF[11] = origin.y;
        this.headerF[12] = origin.z;
        return k;
    }

    placement(pose: WorldTransform, origin: Pos): void {
        this.input[0] = pose.p.x - origin.x;
        this.input[1] = pose.p.y - origin.y;
        this.input[2] = pose.p.z - origin.z;
        this.input[3] = pose.q.v.x;
        this.input[4] = pose.q.v.y;
        this.input[5] = pose.q.v.z;
        this.input[6] = pose.q.s;
    }

    proxy(proxy: ShapeProxy): void {
        this.input[7] = Math.min(proxy.count, 128);
        this.input[8] = proxy.radius;
        for (let i = 0; i < Math.min(proxy.count, 128); ++i) {
            const p = proxy.points[i];
            const n = 14 + 3 * i;
            this.input[n] = p.x;
            this.input[n + 1] = p.y;
            this.input[n + 2] = p.z;
        }
    }
    mover(center1: Vec3, center2: Vec3, radius: number): void {
        this.input[7] = 2;
        this.input[8] = radius;
        this.input[14] = center1.x;
        this.input[15] = center1.y;
        this.input[16] = center1.z;
        this.input[17] = center2.x;
        this.input[18] = center2.y;
        this.input[19] = center2.z;
    }
    translation(v: Vec3): void {
        this.input[9] = v.x;
        this.input[10] = v.y;
        this.input[11] = v.z;
    }
    bounds(box: AABB): void {
        const f = this.headerF;
        f[13] = box.lowerBound.x;
        f[14] = box.lowerBound.y;
        f[15] = box.lowerBound.z;
        f[16] = box.upperBound.x;
        f[17] = box.upperBound.y;
        f[18] = box.upperBound.z;
    }
}
