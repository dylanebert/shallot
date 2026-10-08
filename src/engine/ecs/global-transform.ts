import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { Xform } from "../utils";
import { component, vec4 } from "./component";
import type { System } from "./scheduler";
import type { ComponentStorage } from "./storage";
import type { GpuTable } from "./table";
import type { World } from "./world";

/** Derived fixed-tick world placement, never authored. Gameplay and physics queries
 * read this, never Transform. Producers require it on insertion;
 * rendering reads `globalTransformTable(world)` instead of these fixed-tick columns. */
export const GlobalTransform = component(
    "GlobalTransform",
    {
        translation: vec4,
        rotation: vec4,
        scale: vec4,
        linearVelocity: vec4,
    },
    {
        defaults: () => ({
            translation: [0, 0, 0, 0],
            rotation: [0, 0, 0, 1],
            scale: [1, 1, 1, 0],
            linearVelocity: [0, 0, 0, 0],
        }),
    },
);
/** Authored world placement. There is no hierarchy. */
export const Transform = component(
    "Transform",
    { translation: vec4, rotation: vec4, scale: vec4 },
    {
        defaults: () => ({
            translation: [0, 0, 0, 0],
            rotation: [0, 0, 0, 1],
            scale: [1, 1, 1, 1],
        }),
        requires: [GlobalTransform],
    },
);
const transformTerms = [Transform];
const globalTransformTerms = [GlobalTransform];
const layout = tgpu.bindGroupLayout({
    current: { storage: d.arrayOf(Xform), access: "readonly" },
    previous: { storage: d.arrayOf(Xform), access: "readonly" },
    output: { storage: d.arrayOf(Xform), access: "mutable" },
    rows: { storage: d.arrayOf(d.vec2u), access: "readonly" },
    params: { uniform: d.vec4f },
});
const kernel = tgpu
    .computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })((args) => {
        "use gpu";
        const i = args.gid.x;
        if (i >= d.u32(layout.$.params.y)) return;
        const row = layout.$.rows[i].y;
        const current = layout.$.current[row];
        const previous = layout.$.previous[row];
        const alpha = layout.$.params.x;
        const flip = std.select(d.f32(1), d.f32(-1), std.dot(previous.quat, current.quat) < 0);
        const blend = std.add(
            std.mul(previous.quat, flip * (1 - alpha)),
            std.mul(current.quat, alpha),
        );
        const len = std.length(blend);
        let quat = d.vec4f(0, 0, 0, 1);
        if (len > 1e-12) quat = std.div(blend, len);
        layout.$.output[row] = Xform({
            pos: std.mix(previous.pos, current.pos, alpha),
            quat,
            scale: std.mix(previous.scale, current.scale, alpha),
        });
    })
    .$name("globalTransformInterpolate");

/** @internal World-owned current placement, GPU-only history and interpolated rows. */
export interface GlobalTransformRuntime {
    current?: GpuTable<typeof Xform>;
    previous?: GpuTable<typeof Xform>;
    render?: GpuTable<typeof Xform>;
    enabled: boolean;
    tickCount: number;
    stages: (GPUBuffer | undefined)[];
    ranges: Uint32Array;
    pipeline?: GPUComputePipeline;
    group?: GPUBindGroup;
    generation: number;
    params?: GPUBuffer;
    placement: ComponentStorage<typeof Transform>;
    global: ComponentStorage<typeof GlobalTransform>;
    discontinuities: Uint32Array;
    discontinuityCount: number;
    historyNeedsPromotion: boolean;
}

/** @internal Register the built-in schemas; plugins cannot opt out of world placement. */
export function registerGlobalTransform(world: World): void {
    world.registry.register(GlobalTransform);
    world.registry.register(Transform);
}
/** @internal Install once before setup authoring. GPU residency waits for a reader. */
export function initializeGlobalTransform(world: World): void {
    if (world.globalTransformRuntime) return;
    world.addBoundarySystem(GlobalTransformTickStartSystem, "before");
    world.addBoundarySystem(GlobalTransformTickEndSystem, "after");
    world.addBoundarySystem(PrepareGlobalTransformSystem, "before");
    const runtime: GlobalTransformRuntime = {
        enabled: false,
        tickCount: 0,
        // Only the latest tick pair survives until presentation.
        stages: new Array(2),
        ranges: new Uint32Array(4),
        group: undefined,
        generation: -1,
        placement: world.storage(Transform),
        global: world.storage(GlobalTransform),
        discontinuities: new Uint32Array(1),
        discontinuityCount: 0,
        historyNeedsPromotion: false,
    };
    world.globalTransformRuntime = runtime;
    world.observeMembership(GlobalTransform, (eid, present) => {
        if (present && runtime.enabled) queueDiscontinuity(runtime, eid);
    });
}

/** Runs before every fixed system, including first systems. Installed with world placement. */
export const GlobalTransformTickStartSystem: System = {
    group: "fixed",
    first: true,
    name: "global-transform-tick-start",
    update: beginGlobalTransformTick,
};

/** Runs after every fixed placement writer, including last and terminal systems; no ordering edge is needed. */
export const GlobalTransformTickEndSystem: System = {
    group: "fixed",
    last: true,
    name: "global-transform-tick-end",
    update: endGlobalTransformTick,
};

/** Gathers after simulation and before all draw systems, including the frame encoder and upload point. */
export const PrepareGlobalTransformSystem: System = {
    group: "draw",
    first: true,
    name: "prepare-global-transform",
    update: prepareGlobalTransform,
};

/** The engine's interpolated dense GlobalTransform rows. Request before stepping a renderer. */
export function globalTransformTable(world: World): GpuTable<typeof Xform> {
    const runtime = world.globalTransformRuntime;
    if (!runtime) throw new Error("GlobalTransform is unavailable before engine initialization");
    if (!runtime.enabled) {
        runtime.enabled = true;
        const current = (runtime.current = world.table("global-transform", Xform));
        const previous = (runtime.previous = world.table("global-transform-previous-tick", Xform, {
            gpuOnly: true,
        }));
        const render = (runtime.render = world.table("global-transform-interpolated", Xform, {
            gpuOnly: true,
        }));
        current.bindComponent(GlobalTransform, {
            pos: "translation",
            quat: "rotation",
            scale: "scale",
        });
        previous.bindMembership(GlobalTransform);
        render.bindMembership(GlobalTransform);
        render.subscribe((buffer) => {
            world.gpu.buffers.set("globalTransforms", buffer);
            world.gpu.typed.set("globalTransforms", render.typed);
        });
        render.enableEidLookup();
        render.subscribeMap((buffer) => {
            world.gpu.buffers.set("globalTransformRows", buffer);
            world.gpu.typed.set("globalTransformRows", render.eidToRowTyped!);
        });
        if (!runtime.params) {
            const params = world.gpu.root.createBuffer(d.vec4f).$usage("uniform");
            const buffer = world.gpu.root.unwrap(params);
            runtime.params = buffer;
            world.own(buffer);
        }
        runtime.pipeline = world.gpu.root.unwrap(
            world.gpu.root.createComputePipeline({ compute: kernel }),
        );
        for (const eid of world.query(globalTransformTerms)) queueDiscontinuity(runtime, eid);
    }
    return runtime.render!;
}

function queueDiscontinuity(runtime: GlobalTransformRuntime, eid: number): void {
    for (let i = 0; i < runtime.discontinuityCount; i++) {
        if (runtime.discontinuities[i] === eid) return;
    }
    if (runtime.discontinuityCount === runtime.discontinuities.length) {
        const discontinuities = new Uint32Array(runtime.discontinuities.length * 2);
        discontinuities.set(runtime.discontinuities);
        runtime.discontinuities = discontinuities;
    }
    runtime.discontinuities[runtime.discontinuityCount++] = eid;
}

/** @internal Record a teleport at the latest fixed tick. */
export function markGlobalTransformDiscontinuity(world: World, eid: number): void {
    const runtime = world.globalTransformRuntime;
    if (!runtime?.enabled || !world.has(eid, GlobalTransform)) return;
    queueDiscontinuity(runtime, eid);
}

/** @internal Gather authored placement into the fixed world column without per-row callbacks. */
export function deriveTransforms(world: World): void {
    const runtime = world.globalTransformRuntime;
    if (!runtime) return;
    const global = runtime.global,
        source = runtime.placement;
    const pp = source.translation.column,
        pq = source.rotation.column,
        ps = source.scale.column;
    const op = global.translation.column,
        oq = global.rotation.column,
        os = global.scale.column;
    const pd = world.fieldStorage(GlobalTransform, "translation").dirty,
        qd = world.fieldStorage(GlobalTransform, "rotation").dirty,
        sd = world.fieldStorage(GlobalTransform, "scale").dirty;
    const spd = world.fieldStorage(Transform, "translation").dirty,
        sqd = world.fieldStorage(Transform, "rotation").dirty,
        ssd = world.fieldStorage(Transform, "scale").dirty;
    for (const eid of world.query(transformTerms)) {
        const word = eid >>> 5,
            mask = 1 << (eid & 31);
        if (((spd[word] | sqd[word] | ssd[word]) & mask) === 0) continue;
        const offset = eid * 4;
        let posChanged = false,
            quatChanged = false,
            scaleChanged = false;
        for (let lane = 0; lane < 4; lane++) {
            const j = offset + lane;
            if (!Object.is(op[j], pp[j])) {
                op[j] = pp[j];
                posChanged = true;
            }
            if (!Object.is(oq[j], pq[j])) {
                oq[j] = pq[j];
                quatChanged = true;
            }
            if (!Object.is(os[j], ps[j])) {
                os[j] = ps[j];
                scaleChanged = true;
            }
        }
        if (posChanged) pd[word] |= mask;
        if (quatChanged) qd[word] |= mask;
        if (scaleChanged) sd[word] |= mask;
    }
}

function captureCurrent(world: World, phase: number): void {
    const runtime = world.globalTransformRuntime!;
    const current = runtime.current!;
    current.prepareUpload();
    // Reusing a stage must retain earlier changed rows, including gaps in the range.
    if (current.pendingUploadSize && runtime.ranges[phase * 2 + 1]) {
        current.markRange(
            runtime.ranges[phase * 2] / current.rowBytes,
            runtime.ranges[phase * 2 + 1] / current.rowBytes,
        );
    }
    const size = current.pendingUploadSize;
    let buffer = runtime.stages[phase];
    if (size && (!buffer || buffer.size < size)) {
        if (buffer) world.retireGpuBuffer(buffer);
        buffer = world.gpu.device.createBuffer({
            label: `global-transform-tick-${phase}`,
            size,
            usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        world.own(buffer);
        runtime.stages[phase] = buffer;
    }
    if (size && buffer) {
        current.upload(buffer, true);
        runtime.ranges[phase * 2] = current.lastUploadOffset;
        runtime.ranges[phase * 2 + 1] = size;
    }
}
/** @internal Initial placement precedes this frame's fixed ticks; no reader means no GPU work. */
export function beginGlobalTransformTick(world: World): void {
    deriveTransforms(world);
    const runtime = world.globalTransformRuntime;
    if (runtime?.enabled) {
        if (runtime.tickCount && runtime.ranges[3]) {
            runtime.current!.markRange(
                runtime.ranges[2] / runtime.current!.rowBytes,
                runtime.ranges[3] / runtime.current!.rowBytes,
            );
        }
        captureCurrent(world, 0);
        runtime.discontinuityCount = 0;
    }
}
/** @internal Retain the latest completed tick for presentation. */
export function endGlobalTransformTick(world: World): void {
    deriveTransforms(world);
    const runtime = world.globalTransformRuntime;
    if (runtime?.enabled) {
        runtime.tickCount = 1;
        captureCurrent(world, 1);
    }
}
/** @internal Gather post-simulation placement; recording waits for the renderer's frame encoder. */
export function prepareGlobalTransform(world: World): void {
    deriveTransforms(world);
    const runtime = world.globalTransformRuntime;
    if (!runtime?.enabled) return;
    captureCurrent(world, 1);
    runtime.render!.upload();
}

function copyPhase(world: World, encoder: GPUCommandEncoder, phase: number): void {
    const runtime = world.globalTransformRuntime!;
    const offset = runtime.ranges[phase * 2],
        size = runtime.ranges[phase * 2 + 1];
    if (size && !runtime.stages[phase]) {
        throw new Error(
            `GlobalTransform history phase ${phase} has ${size} bytes but no staging buffer`,
        );
    }
    if (size) {
        encoder.copyBufferToBuffer(
            runtime.stages[phase]!,
            0,
            runtime.current!.buffer,
            offset,
            size,
        );
        runtime.historyNeedsPromotion = true;
    }
    for (let i = 0; i < runtime.discontinuityCount; i++) {
        if (phase !== 1) continue;
        const current = runtime.current!;
        const row = current.rowIndex(runtime.discontinuities[i]);
        if (row >= 0)
            encoder.copyBufferToBuffer(
                current.buffer,
                row * current.rowBytes,
                runtime.previous!.buffer,
                row * current.rowBytes,
                current.rowBytes,
            );
    }
}
/** @internal Record history copies and bind data before renderer compute passes. */
export function prepareGlobalTransformFrame(world: World, encoder: GPUCommandEncoder): void {
    const runtime = world.globalTransformRuntime;
    if (!runtime?.enabled) return;
    if (runtime.tickCount) {
        copyPhase(world, encoder, 0);
        if (runtime.historyNeedsPromotion) {
            encoder.copyBufferToBuffer(
                runtime.current!.buffer,
                0,
                runtime.previous!.buffer,
                0,
                runtime.current!.buffer.size,
            );
            runtime.historyNeedsPromotion = false;
        }
    }
    copyPhase(world, encoder, 1);
    runtime.tickCount = 0;
    runtime.discontinuityCount = 0;
    runtime.ranges.fill(0);
    if (!runtime.current!.count) {
        runtime.historyNeedsPromotion = false;
        return;
    }
    const generation =
        runtime.current!.generation +
        runtime.previous!.generation +
        runtime.render!.generation +
        runtime.current!.activeGeneration;
    if (generation !== runtime.generation) {
        runtime.group = world.gpu.device.createBindGroup({
            layout: runtime.pipeline!.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: runtime.current!.buffer } },
                { binding: 1, resource: { buffer: runtime.previous!.buffer } },
                { binding: 2, resource: { buffer: runtime.render!.buffer } },
                { binding: 3, resource: { buffer: runtime.current!.activeRowsBuffer! } },
                { binding: 4, resource: { buffer: runtime.params! } },
            ],
        });
        runtime.generation = generation;
    }
}

/** Compose this World's fixed-tick GlobalTransform for CPU camera and query readers. */
export function composeGlobalTransform(world: World, eid: number, out: Float32Array): Float32Array {
    const { translation: pos, rotation: rot, scale } = world.storage(GlobalTransform);
    const px = pos.x.get(eid),
        py = pos.y.get(eid),
        pz = pos.z.get(eid);
    const qx = rot.x.get(eid),
        qy = rot.y.get(eid),
        qz = rot.z.get(eid),
        qw = rot.w.get(eid);
    const sx = scale.x.get(eid),
        sy = scale.y.get(eid),
        sz = scale.z.get(eid);
    const x2 = qx + qx,
        y2 = qy + qy,
        z2 = qz + qz;
    const xx = qx * x2,
        xy = qx * y2,
        xz = qx * z2,
        yy = qy * y2,
        yz = qy * z2,
        zz = qz * z2;
    const wx = qw * x2,
        wy = qw * y2,
        wz = qw * z2;
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
