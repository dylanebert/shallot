import type { World } from "../../engine";
import type { GltfMesh } from "./gltf";

/** Contiguous ranges; released neighbours coalesce and the free tail leaves the extent. */
export class DeformationRanges {
    extent = 0;
    private _free: { first: number; count: number }[] = [];
    allocate(count: number, limit: number, name: string): number {
        if (!Number.isInteger(count) || count <= 0)
            throw new Error(`${name} requires a positive count`);
        const hole = this._free.findIndex((range) => range.count >= count);
        if (hole < 0 && this.extent + count > limit)
            throw new Error(`${name} exceeds WebGPU's default texture dimensions`);
        if (hole < 0) {
            const first = this.extent;
            this.extent += count;
            return first;
        }
        const range = this._free[hole],
            first = range.first;
        range.first += count;
        range.count -= count;
        if (!range.count) this._free.splice(hole, 1);
        return first;
    }
    release(first: number, count: number): void {
        this._free.push({ first, count });
        this._free.sort((a, b) => a.first - b.first);
        for (let i = this._free.length - 2; i >= 0; i--) {
            const a = this._free[i],
                b = this._free[i + 1];
            if (a.first + a.count === b.first) {
                a.count += b.count;
                this._free.splice(i + 1, 1);
            }
        }
        const last = this._free.at(-1);
        if (last && last.first + last.count === this.extent) {
            this.extent = last.first;
            this._free.pop();
        }
    }
}

export const MorphWidth = 4096;
export const WeightWidth = 256;

/** World-owned float textures. Registration happens before warm; queue writes preserve prior uploads. */
export class MorphTextures {
    readonly ranges = new DeformationRanges();
    weights = new Float32Array(0);
    private _deltas = new Float32Array(0);
    private _texels = 0;
    private _weightHeight = 0;
    private _deltaHeight = 0;
    readonly bindings = {
        morphWeights: undefined as unknown as GPUTexture,
        morphDeltas: undefined as unknown as GPUTexture,
    };
    private readonly _weightTarget = { texture: undefined as unknown as GPUTexture };
    private readonly _weightLayout = { bytesPerRow: WeightWidth * 4 };
    private readonly _weightSize = { width: WeightWidth, height: 1 };
    private readonly _world: World;
    constructor(world: World) {
        this._world = world;
    }
    private texture(width: number, height: number, format: GPUTextureFormat): GPUTexture {
        return this._world.gpu.device.createTexture({
            label: format === "r32float" ? "glTF morph weights" : "glTF morph deltas",
            size: [width, height],
            format,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
    }
    allocate(count: number): number {
        const first = this.ranges.allocate(count, WeightWidth * 8192, "MorphedMesh weights");
        this.growWeights();
        this.weights.fill(0, first, first + count);
        return first;
    }
    private growWeights(): void {
        const required = Math.max(1, Math.ceil(this.ranges.extent / WeightWidth));
        if (required <= this._weightHeight) return;
        this._weightHeight = Math.min(8192, Math.max(required, this._weightHeight * 2));
        const data = new Float32Array(this._weightHeight * WeightWidth);
        data.set(this.weights);
        this.weights = data;
        if (this.bindings.morphWeights) this._world.retireGpuResource(this.bindings.morphWeights);
        this.bindings.morphWeights = this._weightTarget.texture = this.texture(
            WeightWidth,
            this._weightHeight,
            "r32float",
        );
    }
    register(mesh: GltfMesh): { first: number; mask: number; stride: number } {
        const targets = mesh.morphTargets ?? [];
        let mask = 0;
        for (const target of targets) {
            if (target.position) mask |= 1;
            if (target.normal) mask |= 2;
            if (target.tangent) mask |= 4;
        }
        const stride = (mask & 1) + ((mask >> 1) & 1) + ((mask >> 2) & 1);
        const first = this._texels;
        const end = first + (mesh.vertices.length / 8) * targets.length * stride;
        const required = Math.max(1, Math.ceil(end / MorphWidth));
        if (required > 8192)
            throw new Error(
                `glTF ${mesh.name}: morph deltas exceed WebGPU's default texture dimensions`,
            );
        this.growDeltas(required);
        this._texels = end;
        let texel = first;
        for (let v = 0; v < mesh.vertices.length / 8; v++)
            for (const target of targets)
                for (const [a, semantic] of (
                    ["position", "normal", "tangent"] as const
                ).entries()) {
                    if (!(mask & (1 << a))) continue;
                    const source = target[semantic];
                    for (let k = 0; k < 3; k++)
                        this._deltas[texel * 4 + k] = source?.[v * 3 + k] ?? 0;
                    texel++;
                }
        this._world.gpu.device.queue.writeTexture(
            { texture: this.bindings.morphDeltas },
            this._deltas,
            { bytesPerRow: MorphWidth * 16 },
            { width: MorphWidth, height: required },
        );
        return { first, mask, stride };
    }
    private growDeltas(required: number): void {
        if (required <= this._deltaHeight) return;
        let height = Math.max(1, this._deltaHeight);
        while (height < required) height *= 2;
        this._deltaHeight = height;
        const data = new Float32Array(height * MorphWidth * 4);
        data.set(this._deltas);
        this._deltas = data;
        if (this.bindings.morphDeltas) this._world.retireGpuResource(this.bindings.morphDeltas);
        this.bindings.morphDeltas = this.texture(MorphWidth, height, "rgba32float");
    }
    upload(): void {
        this.growWeights();
        this.growDeltas(1);
        this._weightSize.height = Math.max(1, Math.ceil(this.ranges.extent / WeightWidth));
        this._world.gpu.device.queue.writeTexture(
            this._weightTarget,
            this.weights,
            this._weightLayout,
            this._weightSize,
        );
    }
}
