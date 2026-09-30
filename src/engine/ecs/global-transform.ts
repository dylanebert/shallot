import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { eulerAlias, Xform } from "../utils";
import { type Component, idOf, vec4 } from "./component";
import type { World } from "./state";
import type { ComponentStorage } from "./storage";
import type { GpuTable } from "./table";

/** Authored world placement. A simulated body excludes this producer. There is no hierarchy. */
export const Transform = { translation: vec4, rotation: vec4, scale: vec4 };
/** Derived fixed-tick world placement, never scene-authored. Gameplay and physics queries
 * read this, never Transform. Exactly one component provides it per entity through traits;
 * rendering reads `globalTransformTable(world)` instead of these fixed-tick columns. */
export const GlobalTransform = {
    translation: vec4,
    rotation: vec4,
    scale: vec4,
    linearVelocity: vec4,
};
export const globalTransformTraits = {
    derived: true,
    defaults: () => ({
        translation: [0, 0, 0, 0],
        rotation: [0, 0, 0, 1],
        scale: [1, 1, 1, 0],
        linearVelocity: [0, 0, 0, 0],
    }),
};
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
    captureIndex: number;
    stages: (GPUBuffer | undefined)[];
    ranges: Uint32Array;
    pipeline?: GPUComputePipeline;
    group?: GPUBindGroup;
    generation: number;
    params?: GPUBuffer;
    placement: ComponentStorage<typeof Transform>;
    global: ComponentStorage<typeof GlobalTransform>;
    producers: Map<number, Set<number>>;
    pendingRemoval: Set<number>;
    discontinuities: Uint32Array;
    discontinuityPhases: Uint8Array;
    discontinuityCount: number;
    historyNeedsPromotion: boolean;
}

/** @internal Register the built-in schemas; plugins cannot opt out of world placement. */
export function registerGlobalTransform(world: World): void {
    world.registry.register("GlobalTransform", GlobalTransform, globalTransformTraits);
    world.registry.register("Transform", Transform, {
        defaults: () => ({
            translation: [0, 0, 0, 0],
            rotation: [0, 0, 0, 1],
            scale: [1, 1, 1, 1],
        }),
        aliases: { rotation: eulerAlias("rotation") },
        provides: [GlobalTransform],
    });
}
/** @internal Install once before scene/setup authoring. GPU residency waits for a reader. */
export function initializeGlobalTransform(world: World): void {
    const runtime: GlobalTransformRuntime = {
        enabled: false,
        tickCount: 0,
        captureIndex: 0,
        // Initial placement, four bounded catch-up ticks, and post-simulation placement.
        stages: new Array(6),
        ranges: new Uint32Array(12),
        group: undefined,
        generation: -1,
        placement: world.storage(Transform),
        global: world.storage(GlobalTransform),
        producers: new Map(),
        pendingRemoval: new Set(),
        discontinuities: new Uint32Array(1),
        discontinuityPhases: new Uint8Array(1),
        discontinuityCount: 0,
        historyNeedsPromotion: false,
    };
    world.globalTransformRuntime = runtime;
    world.observeMembership(GlobalTransform, (eid, present) => {
        if (present && runtime.enabled) queueFresh(runtime, eid);
    });
}

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
        for (const eid of world.query(globalTransformTerms)) queueFresh(runtime, eid);
    }
    return runtime.render!;
}

/** @internal Producer membership and derived-component lifetime belong to the engine. */
export function globalTransformProducerChanged(
    world: World,
    component: Component,
    eid: number,
    present: boolean,
): void {
    if (!world.registry.provides(component, GlobalTransform)) return;
    const runtime = world.globalTransformRuntime;
    if (!runtime) return;
    let producers = runtime.producers.get(eid);
    if (present) {
        if (!producers) runtime.producers.set(eid, (producers = new Set()));
        producers.add(idOf(component));
        runtime.pendingRemoval.delete(eid);
        if (!world.has(eid, GlobalTransform)) world.add(eid, GlobalTransform);
    } else {
        producers?.delete(idOf(component));
        if (producers?.size === 0) runtime.pendingRemoval.add(eid);
    }
}

/** @internal A producer cannot remove a derived row still owned by another producer. */
export function retainsGlobalTransform(world: World, eid: number, component: Component): boolean {
    if (component !== GlobalTransform) return false;
    return (world.globalTransformRuntime?.producers.get(eid)?.size ?? 0) > 0;
}

/** @internal Destruction clears owner state along with the entity's component membership. */
export function forgetGlobalTransformEntity(world: World, eid: number): void {
    const runtime = world.globalTransformRuntime;
    runtime?.producers.delete(eid);
    runtime?.pendingRemoval.delete(eid);
}

/** @internal Reconcile one-frame producer gaps before world placement is derived for draw. */
export function reconcileGlobalTransformProducers(world: World): void {
    const runtime = world.globalTransformRuntime;
    if (!runtime) return;
    for (const eid of runtime.pendingRemoval) {
        if (runtime.producers.get(eid)?.size) continue;
        runtime.pendingRemoval.delete(eid);
        runtime.producers.delete(eid);
        if (world.has(eid, GlobalTransform)) world.remove(eid, GlobalTransform);
    }
}

function queueDiscontinuity(runtime: GlobalTransformRuntime, eid: number, phase: number): void {
    for (let i = 0; i < runtime.discontinuityCount; i++) {
        if (runtime.discontinuities[i] === eid && runtime.discontinuityPhases[i] === phase) return;
    }
    if (runtime.discontinuityCount === runtime.discontinuities.length) {
        const discontinuities = new Uint32Array(runtime.discontinuities.length * 2);
        const phases = new Uint8Array(discontinuities.length);
        discontinuities.set(runtime.discontinuities);
        phases.set(runtime.discontinuityPhases);
        runtime.discontinuities = discontinuities;
        runtime.discontinuityPhases = phases;
    }
    runtime.discontinuities[runtime.discontinuityCount] = eid;
    runtime.discontinuityPhases[runtime.discontinuityCount++] = phase;
}

function queueFresh(runtime: GlobalTransformRuntime, eid: number): void {
    queueDiscontinuity(runtime, eid, runtime.captureIndex);
}

/** @internal Record a teleport at the current fixed-history phase. */
export function markGlobalTransformDiscontinuity(world: World, eid: number): void {
    const runtime = world.globalTransformRuntime;
    if (!runtime?.enabled || !world.has(eid, GlobalTransform)) return;
    queueDiscontinuity(runtime, eid, runtime.captureIndex);
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
    if (size && buffer) current.upload(buffer, true);
    runtime.ranges[phase * 2] = size ? current.lastUploadOffset : 0;
    runtime.ranges[phase * 2 + 1] = size;
    runtime.captureIndex = phase + 1;
}
/** @internal Initial placement precedes this frame's fixed ticks; no reader means no GPU work. */
export function beginGlobalTransformTick(world: World): void {
    deriveTransforms(world);
    const runtime = world.globalTransformRuntime;
    if (runtime?.enabled && runtime.tickCount === 0) captureCurrent(world, 0);
}
/** @internal Stage changed current rows; each catch-up tick has distinct immutable GPU bytes. */
export function endGlobalTransformTick(world: World): void {
    deriveTransforms(world);
    const runtime = world.globalTransformRuntime;
    if (runtime?.enabled) captureCurrent(world, ++runtime.tickCount);
}
/** @internal Gather post-simulation placement; recording waits for the renderer's frame encoder. */
export function prepareGlobalTransform(world: World): void {
    reconcileGlobalTransformProducers(world);
    deriveTransforms(world);
    const runtime = world.globalTransformRuntime;
    if (!runtime?.enabled) return;
    captureCurrent(world, runtime.tickCount ? runtime.tickCount + 1 : 0);
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
        if (runtime.discontinuityPhases[i] !== phase) continue;
        const row = runtime.current!.rowIndex(runtime.discontinuities[i]);
        if (row >= 0)
            encoder.copyBufferToBuffer(
                runtime.current!.buffer,
                row * 48,
                runtime.previous!.buffer,
                row * 48,
                48,
            );
    }
}
/** @internal Record history copies and bind data before renderer compute passes. */
export function prepareGlobalTransformFrame(world: World, encoder: GPUCommandEncoder): void {
    const runtime = world.globalTransformRuntime;
    if (!runtime?.enabled) return;
    copyPhase(world, encoder, 0);
    for (let tick = 1; tick <= runtime.tickCount; tick++) {
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
        copyPhase(world, encoder, tick);
    }
    if (runtime.tickCount) copyPhase(world, encoder, runtime.tickCount + 1);
    runtime.tickCount = 0;
    runtime.captureIndex = 0;
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
