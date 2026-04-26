import {
    beginComputePass,
    type ComputeNode,
    type ExecutionContext,
} from "../../../standard/compute";
import {
    binding,
    bindView,
    type Binding,
    type BufferView,
    type GBuf,
} from "../../../standard/compute";
import type { BLASAtlas } from "./blas";
import { AABB_SENTINEL_WGSL, SAFE_INVERSE_EPSILON } from "./structs";

const WORKGROUP_SIZE = 64;

const shader = /* wgsl */ `
${AABB_SENTINEL_WGSL}
const INVALID_SHAPE: u32 = 0xFFFFFFFFu;

struct ShapeAABB {
    minX: f32,
    minY: f32,
    minZ: f32,
    _pad0: u32,
    maxX: f32,
    maxY: f32,
    maxZ: f32,
    _pad1: u32,
}

struct InstanceAABB {
    minX: f32,
    minY: f32,
    minZ: f32,
    _pad0: u32,
    maxX: f32,
    maxY: f32,
    maxZ: f32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> matrices: array<mat4x4<f32>>;
@group(0) @binding(1) var<storage, read> sizes: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> perEntityShapeAABBs: array<ShapeAABB>;
@group(0) @binding(3) var<storage, read> entityCount: array<u32>;
@group(0) @binding(4) var<storage, read_write> instanceAABBs: array<InstanceAABB>;
@group(0) @binding(5) var<storage, read_write> instanceInverses: array<mat4x4<f32>>;
@group(0) @binding(6) var<storage, read> shapes: array<u32>;
@group(0) @binding(7) var<storage, read> shapeData: array<u32>;
@group(0) @binding(8) var<storage, read_write> entityBlasMeta: array<u32>;

fn scaleColumns(m: mat4x4<f32>, s: vec3<f32>) -> mat4x4<f32> {
    return mat4x4<f32>(
        m[0] * s.x,
        m[1] * s.y,
        m[2] * s.z,
        m[3]
    );
}

fn transformPoint(p: vec3<f32>, m: mat4x4<f32>) -> vec3<f32> {
    return (m * vec4<f32>(p, 1.0)).xyz;
}

fn transformAABB(aabbMin: vec3<f32>, aabbMax: vec3<f32>, m: mat4x4<f32>) -> array<vec3<f32>, 2> {
    let corners = array<vec3<f32>, 8>(
        vec3<f32>(aabbMin.x, aabbMin.y, aabbMin.z),
        vec3<f32>(aabbMin.x, aabbMin.y, aabbMax.z),
        vec3<f32>(aabbMin.x, aabbMax.y, aabbMin.z),
        vec3<f32>(aabbMin.x, aabbMax.y, aabbMax.z),
        vec3<f32>(aabbMax.x, aabbMin.y, aabbMin.z),
        vec3<f32>(aabbMax.x, aabbMin.y, aabbMax.z),
        vec3<f32>(aabbMax.x, aabbMax.y, aabbMin.z),
        vec3<f32>(aabbMax.x, aabbMax.y, aabbMax.z)
    );

    var newMin = vec3<f32>(AABB_SENTINEL, AABB_SENTINEL, AABB_SENTINEL);
    var newMax = vec3<f32>(-AABB_SENTINEL, -AABB_SENTINEL, -AABB_SENTINEL);

    for (var i = 0u; i < 8u; i++) {
        let t = transformPoint(corners[i], m);
        newMin = min(newMin, t);
        newMax = max(newMax, t);
    }

    return array<vec3<f32>, 2>(newMin, newMax);
}

fn inverse4x4(m: mat4x4<f32>) -> mat4x4<f32> {
    let m00 = m[0][0]; let m10 = m[0][1]; let m20 = m[0][2]; let m30 = m[0][3];
    let m01 = m[1][0]; let m11 = m[1][1]; let m21 = m[1][2]; let m31 = m[1][3];
    let m02 = m[2][0]; let m12 = m[2][1]; let m22 = m[2][2]; let m32 = m[2][3];
    let m03 = m[3][0]; let m13 = m[3][1]; let m23 = m[3][2]; let m33 = m[3][3];

    let c00 = m11 * (m22 * m33 - m32 * m23) - m21 * (m12 * m33 - m32 * m13) + m31 * (m12 * m23 - m22 * m13);
    let c01 = -(m01 * (m22 * m33 - m32 * m23) - m21 * (m02 * m33 - m32 * m03) + m31 * (m02 * m23 - m22 * m03));
    let c02 = m01 * (m12 * m33 - m32 * m13) - m11 * (m02 * m33 - m32 * m03) + m31 * (m02 * m13 - m12 * m03);
    let c03 = -(m01 * (m12 * m23 - m22 * m13) - m11 * (m02 * m23 - m22 * m03) + m21 * (m02 * m13 - m12 * m03));

    let c10 = -(m10 * (m22 * m33 - m32 * m23) - m20 * (m12 * m33 - m32 * m13) + m30 * (m12 * m23 - m22 * m13));
    let c11 = m00 * (m22 * m33 - m32 * m23) - m20 * (m02 * m33 - m32 * m03) + m30 * (m02 * m23 - m22 * m03);
    let c12 = -(m00 * (m12 * m33 - m32 * m13) - m10 * (m02 * m33 - m32 * m03) + m30 * (m02 * m13 - m12 * m03));
    let c13 = m00 * (m12 * m23 - m22 * m13) - m10 * (m02 * m23 - m22 * m03) + m20 * (m02 * m13 - m12 * m03);

    let c20 = m10 * (m21 * m33 - m31 * m23) - m20 * (m11 * m33 - m31 * m13) + m30 * (m11 * m23 - m21 * m13);
    let c21 = -(m00 * (m21 * m33 - m31 * m23) - m20 * (m01 * m33 - m31 * m03) + m30 * (m01 * m23 - m21 * m03));
    let c22 = m00 * (m11 * m33 - m31 * m13) - m10 * (m01 * m33 - m31 * m03) + m30 * (m01 * m13 - m11 * m03);
    let c23 = -(m00 * (m11 * m23 - m21 * m13) - m10 * (m01 * m23 - m21 * m03) + m20 * (m01 * m13 - m11 * m03));

    let c30 = -(m10 * (m21 * m32 - m31 * m22) - m20 * (m11 * m32 - m31 * m12) + m30 * (m11 * m22 - m21 * m12));
    let c31 = m00 * (m21 * m32 - m31 * m22) - m20 * (m01 * m32 - m31 * m02) + m30 * (m01 * m22 - m21 * m02);
    let c32 = -(m00 * (m11 * m32 - m31 * m12) - m10 * (m01 * m32 - m31 * m02) + m30 * (m01 * m12 - m11 * m02));
    let c33 = m00 * (m11 * m22 - m21 * m12) - m10 * (m01 * m22 - m21 * m02) + m20 * (m01 * m12 - m11 * m02);

    let det = m00 * c00 + m01 * c10 + m02 * c20 + m03 * c30;
    let invDet = select(0.0, 1.0 / det, abs(det) > ${SAFE_INVERSE_EPSILON});

    return mat4x4<f32>(
        vec4<f32>(c00, c10, c20, c30) * invDet,
        vec4<f32>(c01, c11, c21, c31) * invDet,
        vec4<f32>(c02, c12, c22, c32) * invDet,
        vec4<f32>(c03, c13, c23, c33) * invDet
    );
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let eid = gid.x;
    let count = entityCount[0];
    if (eid >= count) { return; }

    let shapeId = shapes[eid];

    if (shapeId == INVALID_SHAPE) {
        var zeroAABB: InstanceAABB;
        zeroAABB.minX = 0.0;
        zeroAABB.minY = 0.0;
        zeroAABB.minZ = 0.0;
        zeroAABB._pad0 = 0u;
        zeroAABB.maxX = 0.0;
        zeroAABB.maxY = 0.0;
        zeroAABB.maxZ = 0.0;
        zeroAABB._pad1 = 0u;
        instanceAABBs[eid] = zeroAABB;
        instanceInverses[eid] = mat4x4<f32>();
        entityBlasMeta[eid * 4u + 0u] = 0u;
        entityBlasMeta[eid * 4u + 1u] = 0u;
        entityBlasMeta[eid * 4u + 2u] = 0u;
        entityBlasMeta[eid * 4u + 3u] = 0u;
        return;
    }

    let base = shapeId * 16u;
    let treeNodeOffset = shapeData[base + 8u];
    let triIdOffset    = shapeData[base + 9u];
    let triOffset      = shapeData[base + 10u];
    let triCount       = shapeData[base + 11u];
    let nodeCount      = shapeData[base + 12u];

    let slot = perEntityShapeAABBs[eid]._pad0;

    entityBlasMeta[eid * 4u + 0u] = treeNodeOffset + slot * nodeCount;
    entityBlasMeta[eid * 4u + 1u] = triIdOffset;
    entityBlasMeta[eid * 4u + 2u] = triOffset + slot * triCount;
    let isAnalytical = shapeId < 4u && slot == 0u;
    entityBlasMeta[eid * 4u + 3u] = select(triCount, 0u, isAnalytical);

    var aabbMin: vec3<f32>;
    var aabbMax: vec3<f32>;
    if (slot > 0u) {
        let dynAABB = perEntityShapeAABBs[eid];
        aabbMin = vec3<f32>(dynAABB.minX, dynAABB.minY, dynAABB.minZ);
        aabbMax = vec3<f32>(dynAABB.maxX, dynAABB.maxY, dynAABB.maxZ);
    } else {
        aabbMin = vec3<f32>(
            bitcast<f32>(shapeData[base + 0u]),
            bitcast<f32>(shapeData[base + 1u]),
            bitcast<f32>(shapeData[base + 2u]));
        aabbMax = vec3<f32>(
            bitcast<f32>(shapeData[base + 4u]),
            bitcast<f32>(shapeData[base + 5u]),
            bitcast<f32>(shapeData[base + 6u]));
    }

    let matrix = matrices[eid];
    let size = sizes[eid].xyz;

    let hasZeroScale = size.x == 0.0 || size.y == 0.0 || size.z == 0.0;

    if (hasZeroScale) {
        var zeroAABB: InstanceAABB;
        zeroAABB.minX = 0.0;
        zeroAABB.minY = 0.0;
        zeroAABB.minZ = 0.0;
        zeroAABB._pad0 = 0u;
        zeroAABB.maxX = 0.0;
        zeroAABB.maxY = 0.0;
        zeroAABB.maxZ = 0.0;
        zeroAABB._pad1 = 0u;
        instanceAABBs[eid] = zeroAABB;
        instanceInverses[eid] = mat4x4<f32>();
        return;
    }

    let scaledMatrix = scaleColumns(matrix, size);
    let worldAABB = transformAABB(aabbMin, aabbMax, scaledMatrix);

    var outAABB: InstanceAABB;
    outAABB.minX = worldAABB[0].x;
    outAABB.minY = worldAABB[0].y;
    outAABB.minZ = worldAABB[0].z;
    outAABB._pad0 = 0u;
    outAABB.maxX = worldAABB[1].x;
    outAABB.maxY = worldAABB[1].y;
    outAABB.maxZ = worldAABB[1].z;
    outAABB._pad1 = 0u;
    instanceAABBs[eid] = outAABB;

    instanceInverses[eid] = inverse4x4(scaledMatrix);
}
`;

export function createInstanceNode(
    bvh: { instanceAABBs: GBuf; instanceInverses: GBuf; blasAtlas: BLASAtlas },
    render: {
        matrices: GBuf;
        sizes: BufferView;
        entityCountBuffer: BufferView;
        shapes: BufferView;
        entityCount: number;
    },
    inputs: string[],
    isActive: () => boolean,
): ComputeNode {
    let pipeline: GPUComputePipeline | null = null;
    let instanceBinding: Binding | null = null;
    let cachedPerEntityAABBs: GPUBuffer | null = null;
    let cachedShapeData: GPUBuffer | null = null;

    return {
        name: "instance",
        scope: "frame",
        inputs,
        outputs: ["instance-aabbs", "instance-inverses"],

        async prepare(device: GPUDevice) {
            const module = device.createShaderModule({ code: shader });

            pipeline = await device.createComputePipelineAsync({
                label: "rt-instance",
                layout: "auto",
                compute: { module, entryPoint: "main" },
            });

            instanceBinding = binding(device, pipeline.getBindGroupLayout(0), () => [
                { binding: 0, resource: { buffer: render.matrices.buffer } },
                bindView(1, render.sizes),
                { binding: 2, resource: { buffer: bvh.blasAtlas.perEntityShapeAABBs } },
                bindView(3, render.entityCountBuffer),
                { binding: 4, resource: { buffer: bvh.instanceAABBs.buffer } },
                { binding: 5, resource: { buffer: bvh.instanceInverses.buffer } },
                bindView(6, render.shapes),
                { binding: 7, resource: { buffer: bvh.blasAtlas.shapeDataBuffer } },
                { binding: 8, resource: { buffer: bvh.blasAtlas.entityBlasMetaBuffer } },
            ]);
            cachedPerEntityAABBs = bvh.blasAtlas.perEntityShapeAABBs;
            cachedShapeData = bvh.blasAtlas.shapeDataBuffer;
        },

        execute(ctx: ExecutionContext) {
            if (!isActive() || !instanceBinding) return;

            if (
                bvh.blasAtlas.perEntityShapeAABBs !== cachedPerEntityAABBs ||
                bvh.blasAtlas.shapeDataBuffer !== cachedShapeData
            ) {
                cachedPerEntityAABBs = bvh.blasAtlas.perEntityShapeAABBs;
                cachedShapeData = bvh.blasAtlas.shapeDataBuffer;
                instanceBinding.invalidate();
            }

            const workgroups = Math.ceil(render.entityCount / WORKGROUP_SIZE);

            const pass = beginComputePass(ctx.encoder, ctx.timestampWrites?.("bvh-instance"));
            pass.setPipeline(pipeline!);
            pass.setBindGroup(0, instanceBinding.group);
            pass.dispatchWorkgroups(workgroups);
            pass.end();
        },
    };
}
