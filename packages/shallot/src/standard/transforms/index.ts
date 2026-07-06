import { Compute, capacity, type Plugin, vec4 } from "../../engine";
import { eulerAlias } from "../../engine/utils";
import { XFORM_WGSL } from "../../engine/utils/core";
import { SlabPlugin, slab } from "../slab";

// the transform firehose: one capacity-sized buffer of decomposed per-entity {pos, quat, scale} (`Xform`,
// 48 B) the compose pass gathers from the pos/rot/scale slabs; readers reconstruct the world transform on
// read (`XFORM_WGSL`), keeping the per-instance read one AoS cache line. Published to `Compute.buffers` as
// "transforms" (the access path — surfaces resolve it by name). A derived GPU buffer, not a per-entity
// field, so it lives here, not on the Transform component (mirrors `Lighting` vs `DirectionalLight` in
// render/). null until initialize (headless: stays null).
let _transforms: GPUBuffer | null = null;
let _composeLayout: GPUBindGroupLayout | null = null;
let _composePipeline: GPUComputePipeline | null = null;
let _composeBindGroup: GPUBindGroup | null = null;

/**
 * per-entity transform: pos, rot, scale as direct {@link Quad} fields. Lanes
 * are individually addressable as Singles (`Transform.pos.x.set(eid, v)`) and
 * writable in bulk (`Transform.pos.set(eid, x, y, z, 0)`). The CPU writes flow
 * through the slab plugin; SlabSystem flushes dirty slots once per frame and
 * the compose pass derives world matrices into the `"transforms"` firehose.
 *
 * @example
 * ```
 * <a transform pos="0 1 0" rot="0 0 0 1" scale="1 1 1" />
 * ```
 */
export const Transform = {
    pos: slab(vec4),
    rot: slab(vec4),
    scale: slab(vec4),
};

// gather the pos/rot/scale slabs into the decomposed `Xform` firehose. No matrix math — readers
// reconstruct on demand (XFORM_WGSL); the gather just lays the three SoA slabs into one AoS record so a
// reader's per-instance read is a single cache line.
const COMPOSE_WGSL =
    XFORM_WGSL +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> rot: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> scale: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> transforms: array<Xform>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&transforms)) { return; }
    transforms[i] = Xform(pos[i].xyz, rot[i], scale[i].xyz);
}
`;

/**
 * record the per-frame world-matrix compose dispatch onto `encoder`. Reads
 * the slab canonical GPU buffers (populated by the prior frame's SlabSystem
 * submit), writes the `"transforms"` firehose. The bind group is built
 * lazily on the first call so it can reference the slab `.gpu` buffers
 * allocated during SlabPlugin's warm
 */
export function composeTransforms(encoder: GPUCommandEncoder): void {
    if (!_composePipeline || !_composeLayout || !_transforms) return;
    if (!_composeBindGroup) {
        _composeBindGroup = Compute.device.createBindGroup({
            label: "kitchen-transforms-compose",
            layout: _composeLayout,
            entries: [
                { binding: 0, resource: { buffer: Transform.pos.gpu! } },
                { binding: 1, resource: { buffer: Transform.rot.gpu! } },
                { binding: 2, resource: { buffer: Transform.scale.gpu! } },
                { binding: 3, resource: { buffer: _transforms } },
            ],
        });
    }
    const pass = encoder.beginComputePass({
        label: "kitchen-transforms-compose",
        timestampWrites: Compute.span?.("transforms:compose"),
    });
    pass.setPipeline(_composePipeline);
    pass.setBindGroup(0, _composeBindGroup);
    pass.dispatchWorkgroups(Math.ceil(capacity / 64));
    pass.end();
}

/**
 * compose one entity's world matrix on CPU from its slab fields. For camera
 * view derivation and other low-count CPU consumers — per-frame entity loops
 * should read the `"transforms"` firehose on GPU instead
 *
 * @example
 * const world = composeTransform(eid, new Float32Array(16));
 */
export function composeTransform(eid: number, out: Float32Array): Float32Array {
    const { pos, rot, scale } = Transform;

    const px = pos.x.get(eid);
    const py = pos.y.get(eid);
    const pz = pos.z.get(eid);
    const qx = rot.x.get(eid);
    const qy = rot.y.get(eid);
    const qz = rot.z.get(eid);
    const qw = rot.w.get(eid);
    const sx = scale.x.get(eid);
    const sy = scale.y.get(eid);
    const sz = scale.z.get(eid);

    const x2 = qx + qx;
    const y2 = qy + qy;
    const z2 = qz + qz;
    const xx = qx * x2;
    const xy = qx * y2;
    const xz = qx * z2;
    const yy = qy * y2;
    const yz = qy * z2;
    const zz = qz * z2;
    const wx = qw * x2;
    const wy = qw * y2;
    const wz = qw * z2;

    out[0] = (1 - yy - zz) * sx;
    out[1] = (xy + wz) * sx;
    out[2] = (xz - wy) * sx;
    out[3] = 0;
    out[4] = (xy - wz) * sy;
    out[5] = (1 - xx - zz) * sy;
    out[6] = (yz + wx) * sy;
    out[7] = 0;
    out[8] = (xz + wy) * sz;
    out[9] = (yz - wx) * sz;
    out[10] = (1 - xx - yy) * sz;
    out[11] = 0;
    out[12] = px;
    out[13] = py;
    out[14] = pz;
    out[15] = 1;
    return out;
}

/**
 * the transform substrate: registers the {@link Transform} component and runs the per-frame compose pass
 * that gathers the pos/rot/scale slabs into the `"transforms"` GPU firehose surfaces read by name. In
 * `DEFAULT_PLUGINS`; every rendered entity needs a Transform.
 */
export const TransformsPlugin: Plugin = {
    name: "Transforms",
    components: { Transform },
    dependencies: [SlabPlugin],
    traits: {
        Transform: {
            defaults: () => ({
                pos: [0, 0, 0, 0],
                rot: [0, 0, 0, 1],
                scale: [1, 1, 1, 1],
            }),
            // rot is stored as a quaternion but authored as euler degrees
            aliases: { rot: eulerAlias("rot") },
        },
    },

    async initialize() {
        _transforms = null;
        _composeLayout = null;
        _composePipeline = null;
        _composeBindGroup = null;

        if (!Compute.device) return;
        const { device } = Compute;

        _transforms = device.createBuffer({
            label: "kitchen-transforms",
            size: capacity * 48, // sizeof(Xform): pos + quat + scale, vec4-aligned
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        Compute.buffers.set("transforms", _transforms);
        _composeLayout = device.createBindGroupLayout({
            label: "kitchen-transforms-compose",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" },
                },
            ],
        });
        const module = device.createShaderModule({
            label: "kitchen-transforms-compose",
            code: COMPOSE_WGSL,
        });
        _composePipeline = await device.createComputePipelineAsync({
            label: "kitchen-transforms-compose",
            layout: device.createPipelineLayout({ bindGroupLayouts: [_composeLayout] }),
            compute: { module, entryPoint: "main" },
        });
    },
};
