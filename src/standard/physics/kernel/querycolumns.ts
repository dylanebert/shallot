import type { ShapeProxy } from "../collision/distance";
import { DEFAULT_CATEGORY_BITS, DEFAULT_MASK_BITS } from "../common/constants";
import type { AABB, Pos, Vec3, WorldTransform } from "../common/records";
import type { QueryFilter } from "../common/types";
import type { WorldState } from "../world/world";
import { assertQueryWorld, type Kernel, kernel } from "./kernel";
import { KernelViews } from "./views";

export function queryColumns(world: WorldState): QueryColumns {
    return (world.queryColumns ??= new QueryColumns(world));
}

/** World-owned views and input upload for the kernel query ABI. */
export class QueryColumns extends KernelViews {
    // Queries leave no logical state between calls; these columns are shared scratch.
    captureCheckpoint(): null {
        return null;
    }
    restoreCheckpoint(_state: unknown): void {}

    readonly world: WorldState;
    headerU = new Uint32Array(0);
    headerF = new Float32Array(0);
    input = new Float32Array(0);
    resultU = new Uint32Array(0);
    resultF = new Float32Array(0);
    cast = new Float32Array(0);
    private readonly _filterBits = new DataView(new ArrayBuffer(8));

    constructor(world: WorldState) {
        super(world.ecsState);
        this.world = world;
        this.guardViews();
    }

    protected deriveViews(): void {
        const k = kernel(this.ecsState);
        const memory = k.memory.buffer;
        if (this.input.buffer !== memory || this.input.byteLength === 0) {
            this.input = new Float32Array(memory, k.shapeQueryInputPtr(), 398);
            this.cast = new Float32Array(memory, k.shapeQueryOutputPtr(), 12);
            this.headerU = new Uint32Array(memory, k.worldQueryHeaderPtr(), 20);
            this.headerF = new Float32Array(memory, k.worldQueryHeaderPtr(), 20);
            this.resultU = new Uint32Array(memory, k.worldQueryResultPtr(), 80);
            this.resultF = new Float32Array(memory, k.worldQueryResultPtr(), 80);
        }
    }

    prepare(origin: Pos, filter?: QueryFilter): Kernel {
        const world = this.world;
        assertQueryWorld(world.ecsState, world.worldId);
        world.broadPhase.store.initialize();
        const k = kernel(world.ecsState);
        k.shapeSetActiveWorld(world.worldId);
        const category = filter?.categoryBits ?? DEFAULT_CATEGORY_BITS;
        const mask = filter?.maskBits ?? DEFAULT_MASK_BITS;
        const x = origin.x;
        const y = origin.y;
        const z = origin.z;
        const h = this.headerU;
        h[19] = 0;
        this._filterBits.setBigUint64(0, category);
        h[6] = this._filterBits.getUint32(0);
        h[7] = this._filterBits.getUint32(4);
        this._filterBits.setBigUint64(0, mask);
        h[8] = this._filterBits.getUint32(0);
        h[9] = this._filterBits.getUint32(4);
        const f = this.headerF;
        f[10] = x;
        f[11] = y;
        f[12] = z;
        return k;
    }

    placement(pose: WorldTransform, origin: Pos, offset = 0): void {
        const px = pose.p.x - origin.x;
        const py = pose.p.y - origin.y;
        const pz = pose.p.z - origin.z;
        const qx = pose.q.v.x;
        const qy = pose.q.v.y;
        const qz = pose.q.v.z;
        const qs = pose.q.s;
        const input = this.input;
        input[offset] = px;
        input[offset + 1] = py;
        input[offset + 2] = pz;
        input[offset + 3] = qx;
        input[offset + 4] = qy;
        input[offset + 5] = qz;
        input[offset + 6] = qs;
    }

    proxy(proxy: ShapeProxy): void {
        const count = Math.min(proxy.count, 128);
        const radius = proxy.radius;
        if (count === 0) {
            const input = this.input;
            input[7] = count;
            input[8] = radius;
        }
        for (let i = 0; i < count; ++i) {
            const p = proxy.points[i];
            const x = p.x;
            const y = p.y;
            const z = p.z;
            const input = this.input;
            const n = 14 + 3 * i;
            input[n] = x;
            input[n + 1] = y;
            input[n + 2] = z;
            input[7] = count;
            input[8] = radius;
        }
    }
    mover(center1: Vec3, center2: Vec3, radius: number): void {
        const ax = center1.x;
        const ay = center1.y;
        const az = center1.z;
        const bx = center2.x;
        const by = center2.y;
        const bz = center2.z;
        const input = this.input;
        input[7] = 2;
        input[8] = radius;
        input[14] = ax;
        input[15] = ay;
        input[16] = az;
        input[17] = bx;
        input[18] = by;
        input[19] = bz;
    }
    translation(v: Vec3): void {
        const x = v.x;
        const y = v.y;
        const z = v.z;
        const input = this.input;
        input[9] = x;
        input[10] = y;
        input[11] = z;
    }
    bounds(box: AABB): void {
        const lx = box.lowerBound.x;
        const ly = box.lowerBound.y;
        const lz = box.lowerBound.z;
        const ux = box.upperBound.x;
        const uy = box.upperBound.y;
        const uz = box.upperBound.z;
        const f = this.headerF;
        f[13] = lx;
        f[14] = ly;
        f[15] = lz;
        f[16] = ux;
        f[17] = uy;
        f[18] = uz;
    }
}
