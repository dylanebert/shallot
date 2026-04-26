export type { Vec3, Ray } from "../../../engine";
import type { Vec3 } from "../../../engine";

export {
    TREE_NODE_SIZE,
    MORTON_QUANTIZATION,
    MAX_PROPAGATION_ITERS,
    LEAF_FLAG,
    AABB_SENTINEL,
    TREE_NODE_STRUCT_WGSL,
    LEAF_FLAG_WGSL,
    AABB_SENTINEL_WGSL,
} from "../../../standard/bvh";

export interface AABB {
    min: Vec3;
    max: Vec3;
}

export interface Triangle {
    v0: Vec3;
    e1: Vec3;
    e2: Vec3;
    n0: Vec3;
    n1: Vec3;
    n2: Vec3;
    entityId: number;
}

export interface BVHNode {
    min: Vec3;
    max: Vec3;
    leftChild: number;
    rightChild: number;
}

export interface MortonPair {
    code: number;
    triangleId: number;
}

export interface HitResult {
    hit: boolean;
    t: number;
    entityId: number;
    u: number;
    v: number;
    normal: Vec3;
    worldPos: Vec3;
}

export function isLeaf(child: number): boolean {
    return (child & 0x80000000) !== 0;
}

export function leafIndex(child: number): number {
    return child & ~0x80000000;
}

export const BVH_NODE_STRUCT_WGSL = /* wgsl */ `
struct BVHNode {
    c0_minX: f32, c0_minY: f32, c0_minZ: f32, child0: u32,
    c0_maxX: f32, c0_maxY: f32, c0_maxZ: f32, _pad0: u32,
    c1_minX: f32, c1_minY: f32, c1_minZ: f32, child1: u32,
    c1_maxX: f32, c1_maxY: f32, c1_maxZ: f32, _pad1: u32,
    c2_minX: f32, c2_minY: f32, c2_minZ: f32, child2: u32,
    c2_maxX: f32, c2_maxY: f32, c2_maxZ: f32, _pad2: u32,
    c3_minX: f32, c3_minY: f32, c3_minZ: f32, child3: u32,
    c3_maxX: f32, c3_maxY: f32, c3_maxZ: f32, _pad3: u32,
}`;

export const BVH_NODE_SIZE = 128;
export const BLAS_TRIANGLE_SIZE = 64;
export const TREE_NODE_STRIDE = 8;
export const BLAS_META_STRIDE = 4;
export const SHAPE_AABB_STRIDE = 8;
export const RAY_EPSILON = 1e-7;
export const SAFE_INVERSE_EPSILON = 1e-10;
export const OCT_ENCODING_SCALE = 65535;

export const BLAS_NODE_STRUCT_WGSL = /* wgsl */ `
struct BLASNode {
    minX: f32, minY: f32, minZ: f32, leftChild: u32,
    maxX: f32, maxY: f32, maxZ: f32, rightChild: u32,
}`;
export const INVALID_NODE = 0xffffffff;

export const BLAS_TRIANGLE_STRUCT_WGSL = /* wgsl */ `
struct BLASTriangle {
    v0: vec3<f32>, _pad0: u32,
    e1: vec3<f32>, _pad1: u32,
    e2: vec3<f32>, _pad2: u32,
    n0_enc: u32, n1_enc: u32, n2_enc: u32, _pad3: u32,
}`;

export const RAY_STRUCT_WGSL = /* wgsl */ `
struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}`;

export const HIT_RESULT_STRUCT_WGSL = /* wgsl */ `
struct HitResult {
    hit: bool,
    t: f32,
    entityId: u32,
    u: f32,
    v: f32,
    normal: vec3<f32>,
    worldPos: vec3<f32>,
}`;

export const OCT_DECODE_WGSL = /* wgsl */ `
fn octDecode(enc: u32) -> vec3<f32> {
    let x = f32(enc & 0xFFFFu) / 65535.0 * 2.0 - 1.0;
    let y = f32(enc >> 16u) / 65535.0 * 2.0 - 1.0;
    let z = 1.0 - abs(x) - abs(y);
    var n: vec3<f32>;
    if (z < 0.0) {
        let signX = select(-1.0, 1.0, x >= 0.0);
        let signY = select(-1.0, 1.0, y >= 0.0);
        n = vec3<f32>((1.0 - abs(y)) * signX, (1.0 - abs(x)) * signY, z);
    } else {
        n = vec3<f32>(x, y, z);
    }
    return normalize(n);
}`;
