import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import type { Plugin } from "../../engine";
import {
    type Component,
    component,
    type GpuTable,
    type Resource,
    type System,
    type World,
} from "../../engine/ecs";
import { composeMat4 } from "../../engine/utils";
import {
    GlobalTransform,
    GlobalTransformTickEndSystem,
    GlobalTransformTickStartSystem,
    PrepareGlobalTransformSystem,
    resetInterpolation,
    TransformPlugin,
    TransformRuntime,
} from "../transform";
import {
    correctRenderTransform,
    prepareVisualCorrection,
    uploadVisualCorrections,
} from "./visual-correction";
import { Xform } from "./xform";

/** Enrolls CPU rendered-placement history. Add before the first fixed sample a CPU
 * reader needs; remove when that explicit request ends. Automatic readers own separate
 * retainRenderTransformHistory leases and never add or remove this marker.
 * Without enrollment, readRenderGlobalTransform reads current placement plus correction. */
export const RenderTransformHistory = component("RenderTransformHistory", {});
const terms = [GlobalTransform];
const cpuTerms = [GlobalTransform, RenderTransformHistory];
type CpuHistory = {
    eid: number;
    index: number;
    previous: Float32Array;
    current: Float32Array;
    captured: boolean;
};
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
/** @internal Rendering-owned fixed samples and GPU placement interpolation. */
export interface GlobalTransformRuntime {
    current?: GpuTable<typeof Xform>;
    previous?: GpuTable<typeof Xform>;
    render?: GpuTable<typeof Xform>;
    enabled: boolean;
    tickCount: number;
    stages: (GPUBuffer | undefined)[];
    cpuPrepared: boolean;
    cpuHistory: Map<number, CpuHistory>;
    cpuRequests: Map<number, { count: number }>;
    cpuReaders: CpuHistory[];
    scratch: Float32Array;
    source: Float32Array;
    ranges: Uint32Array;
    pipeline?: GPUComputePipeline;
    group?: GPUBindGroup;
    generation: number;
    params?: GPUBuffer;
    historyNeedsPromotion: boolean;
    historyPromotionOnSubmit: boolean;
    historyDiscontinuitiesOnSubmit: Uint32Array;
    historyDiscarded: boolean;
    fixedTickRevision: number;
    commitOnSubmit: () => void;
}
/** @internal Sparse CPU samples and GlobalTransform's GPU history, with pending promotion and discontinuities. */
export const GlobalTransformHistory: Resource<GlobalTransformRuntime> = {
    create(world) {
        let runtime: GlobalTransformRuntime;
        runtime = {
            enabled: false,
            tickCount: 0,
            stages: new Array(2),
            cpuPrepared: false,
            cpuHistory: new Map(),
            cpuRequests: new Map(),
            cpuReaders: [],
            scratch: new Float32Array(12),
            source: new Float32Array(0),
            ranges: new Uint32Array(4),
            generation: -1,
            historyNeedsPromotion: false,
            historyPromotionOnSubmit: false,
            historyDiscontinuitiesOnSubmit: new Uint32Array(1),
            historyDiscarded: false,
            fixedTickRevision: world.time.fixedTickRevision,
            commitOnSubmit: () => {
                if (!runtime.enabled) return;
                runtime.tickCount = 0;
                runtime.cpuPrepared = false;
                runtime.ranges.fill(0);
                runtime.historyNeedsPromotion = runtime.historyPromotionOnSubmit;
                runtime.historyDiscontinuitiesOnSubmit.fill(0);
                runtime.historyDiscarded = false;
            },
        };
        const enroll = (eid: number) => syncCpuHistory(world, runtime, eid);
        world.observeMembership(GlobalTransform, (eid, present) => {
            if (!present) runtime.cpuRequests.delete(eid);
            enroll(eid);
            if (present && runtime.enabled) resetInterpolation(world, eid);
        });
        world.observeMembership(RenderTransformHistory, enroll);
        for (const eid of world.query(cpuTerms)) enroll(eid);
        return runtime;
    },
};
function syncCpuHistory(world: World, runtime: GlobalTransformRuntime, eid: number): void {
    const requested = world.has(eid, RenderTransformHistory) || runtime.cpuRequests.has(eid);
    if (world.has(eid, GlobalTransform) && requested) {
        if (!runtime.cpuHistory.has(eid)) {
            const history = {
                eid,
                index: runtime.cpuReaders.length,
                previous: new Float32Array(12),
                current: new Float32Array(12),
                captured: false,
            };
            runtime.cpuHistory.set(eid, history);
            runtime.cpuReaders.push(history);
        }
    } else {
        const history = runtime.cpuHistory.get(eid);
        if (history) {
            const last = runtime.cpuReaders[runtime.cpuReaders.length - 1]!;
            runtime.cpuReaders[history.index] = last;
            last.index = history.index;
            runtime.cpuReaders.pop();
            runtime.cpuHistory.delete(eid);
        }
    }
}
/** Retains CPU fixed samples for a live GlobalTransform before their first rendered read.
 * Each request has its own idempotent release function. The last release frees samples
 * unless an explicit RenderTransformHistory marker remains. Removing GlobalTransform
 * ends all leases; a later placement or reused eid needs a new request. Requires
 * RenderTransformHistoryPlugin (included by RenderingPlugin), and no GPU. */
export function retainRenderTransformHistory(world: World, eid: number): () => void {
    if (!world.has(eid, GlobalTransform))
        throw new Error("retainRenderTransformHistory requires GlobalTransform");
    const runtime = world.resource(GlobalTransformHistory);
    let request = runtime.cpuRequests.get(eid);
    if (!request) {
        request = { count: 0 };
        runtime.cpuRequests.set(eid, request);
    }
    request.count++;
    syncCpuHistory(world, runtime, eid);
    let active = true;
    return () => {
        if (!active) return;
        active = false;
        if (runtime.cpuRequests.get(eid) !== request) return;
        if (--request.count === 0) runtime.cpuRequests.delete(eid);
        syncCpuHistory(world, runtime, eid);
    };
}
/** @internal A component owns its lease only while it and GlobalTransform are present. */
export function observeRenderTransformHistory(world: World, component: Component): void {
    const requests = new Map<number, () => void>();
    const update = (eid: number) => {
        if (world.has(eid, component) && world.has(eid, GlobalTransform)) {
            if (!requests.has(eid)) requests.set(eid, retainRenderTransformHistory(world, eid));
        } else {
            requests.get(eid)?.();
            requests.delete(eid);
        }
    };
    const stopComponent = world.observeMembership(component, update);
    const stopPlacement = world.observeMembership(GlobalTransform, update);
    for (const eid of world.query([component])) update(eid);
    world.onDispose(() => {
        stopComponent();
        stopPlacement();
        for (const release of requests.values()) release();
        requests.clear();
    });
}
/** Captures initial derived placement before every ordinary fixed system. */
export const GlobalTransformHistoryStartSystem: System = {
    group: "fixed",
    name: "global-transform-history-start",
    boundary: "before",
    after: [GlobalTransformTickStartSystem],
    update(world) {
        const runtime = world.resource(GlobalTransformHistory);
        discardJumpedHistory(world, runtime);
        captureCpuHistory(world, runtime, true);
        if (!runtime.enabled) return;
        if (runtime.tickCount && runtime.ranges[3])
            runtime.current!.markRange(
                runtime.ranges[2] / runtime.current!.rowBytes,
                runtime.ranges[3] / runtime.current!.rowBytes,
            );
        captureCurrent(world, 0);
        // These rows are now carried by phase 0, not the earlier phase 1.
        runtime.ranges[2] = runtime.ranges[3] = 0;
        if (runtime.historyDiscarded) runtime.historyNeedsPromotion = true;
        runtime.historyDiscarded = false;
        world.resource(TransformRuntime).discontinuities.fill(0);
        runtime.historyDiscontinuitiesOnSubmit.fill(0);
    },
};
/** Captures completed derived placement after every ordinary fixed writer. */
export const GlobalTransformHistoryEndSystem: System = {
    group: "fixed",
    name: "global-transform-history-end",
    boundary: "after",
    after: [GlobalTransformTickEndSystem],
    update(world) {
        const runtime = world.resource(GlobalTransformHistory);
        captureCpuHistory(world, runtime, false);
        if (runtime.enabled) {
            runtime.tickCount = 1;
            captureCurrent(world, 1);
        }
    },
};
/** Stages simulation placement before ordinary draw systems use the World's encoder. */
export const PrepareGlobalTransformHistorySystem: System = {
    group: "draw",
    name: "prepare-global-transform-history",
    boundary: "before",
    after: [PrepareGlobalTransformSystem],
    update(world) {
        const runtime = world.resource(GlobalTransformHistory);
        if (!runtime.enabled) return;
        discardJumpedHistory(world, runtime);
        markFrameWrites(world, runtime);
        captureCurrent(world, 1);
        runtime.render!.upload();
    },
};
/** @internal Discards presentation history after restoration of fixed placement. */
export function recoverGlobalTransformHistory(world: World) {
    const runtime = world.resource(GlobalTransformHistory);
    return {
        snapshot: () => undefined,
        restore() {
            const current = runtime.current;
            // Staged rows have left the CPU mirror, so keep them pending for upload.
            if (current)
                for (let phase = 0; phase < 2; phase++)
                    if (runtime.ranges[phase * 2 + 1])
                        current.markRange(
                            runtime.ranges[phase * 2] / current.rowBytes,
                            runtime.ranges[phase * 2 + 1] / current.rowBytes,
                        );
            runtime.tickCount = 0;
            runtime.ranges.fill(0);
            runtime.historyNeedsPromotion = false;
            runtime.historyPromotionOnSubmit = false;
            runtime.historyDiscontinuitiesOnSubmit.fill(0);
            runtime.historyDiscarded = true;
        },
    };
}
/** Rendering-owned interpolated dense placement rows. Requires RenderingPlugin;
 * history and interpolation record through `world.frameEncoder()` before draw passes. */
export function globalTransformTable(world: World): GpuTable<typeof Xform> {
    const runtime = world.resource(GlobalTransformHistory);
    if (!runtime.enabled && !world.hasSystem(GlobalTransformHistoryEndSystem))
        throw new Error("globalTransformTable requires RenderingPlugin");
    if (!runtime.enabled) {
        runtime.enabled = true;
        prepareVisualCorrection(world);
        const current = (runtime.current = world.table("global-transform", Xform));
        runtime.previous = world.table("global-transform-previous-tick", Xform, {
            gpuOnly: true,
            rowSource: current,
        });
        const render = (runtime.render = world.table("global-transform-interpolated", Xform, {
            gpuOnly: true,
            rowSource: current,
        }));
        current.bindComponent(GlobalTransform, {
            pos: "translation",
            quat: "rotation",
            scale: "scale",
        });
        render.enableEidLookup();
        if (!runtime.params) {
            const params = world.gpu.root.createBuffer(d.vec4f).$usage("uniform");
            runtime.params = world.gpu.root.unwrap(params);
            world.own(runtime.params);
        }
        runtime.pipeline = world.gpu.root.unwrap(
            world.gpu.root.createComputePipeline({ compute: kernel }),
        );
        for (const eid of world.query(terms)) {
            resetInterpolation(world, eid);
        }
    }
    return runtime.render!;
}
/** The CPU source of rendered entity placement: a caller-owned 12-float Xform record
 * (position at 0, xyzw quaternion at 4, scale at 8), including VisualCorrection.
 * RenderTransformHistory or retainRenderTransformHistory enrolls fixed samples ahead
 * of the first read; the history plugin
 * works without a GPU. Without enrollment, before capture, or after a frame-authored
 * write, reads use current placement. No GPU readback is performed. */
export function readRenderGlobalTransform(world: World, eid: number, out: Float32Array): void {
    const runtime = world.resource(GlobalTransformHistory);
    const { translation, rotation, scale } = world.storage(GlobalTransform);
    const j = eid * 4;
    const p = translation.column,
        q = rotation.column,
        s = scale.column;
    prepareCpuFrame(world, runtime);
    const history = runtime.cpuHistory.get(eid);
    const source = history?.current;
    const frameWritten =
        !source ||
        p[j] !== source[0] ||
        p[j + 1] !== source[1] ||
        p[j + 2] !== source[2] ||
        q[j] !== source[4] ||
        q[j + 1] !== source[5] ||
        q[j + 2] !== source[6] ||
        q[j + 3] !== source[7] ||
        s[j] !== source[8] ||
        s[j + 1] !== source[9] ||
        s[j + 2] !== source[10];
    if (!history?.captured || frameWritten) {
        out[0] = p[j] ?? 0;
        out[1] = p[j + 1] ?? 0;
        out[2] = p[j + 2] ?? 0;
        out[3] = out[11] = 0;
        out[4] = q[j] ?? 0;
        out[5] = q[j + 1] ?? 0;
        out[6] = q[j + 2] ?? 0;
        out[7] = q[j + 3] ?? 0;
        out[8] = s[j] ?? 0;
        out[9] = s[j + 1] ?? 0;
        out[10] = s[j + 2] ?? 0;
        correctRenderTransform(world, eid, out);
        return;
    }
    const offset = 0;
    const current = history.current,
        previous = history.previous;
    const marks = world.resource(TransformRuntime).discontinuities;
    if (marks[eid >>> 5] & (1 << (eid & 31))) previous.set(current);
    const alpha = world.time.fixedAlpha;
    for (let i = 0; i < 12; i++)
        out[i] = previous[offset + i]! * (1 - alpha) + current[offset + i]! * alpha;
    let dot = 0;
    for (let i = 4; i < 8; i++) dot += previous[offset + i]! * current[offset + i]!;
    const flip = dot < 0 ? -1 : 1;
    let length = 0;
    for (let i = 4; i < 8; i++) {
        const value = previous[offset + i]! * flip * (1 - alpha) + current[offset + i]! * alpha;
        out[i] = value;
        length += value * value;
    }
    length = Math.sqrt(length);
    for (let i = 4; i < 8; i++) out[i] = length > 1e-12 ? out[i]! / length : i === 7 ? 1 : 0;
    correctRenderTransform(world, eid, out);
}
/** Composes the CPU rendered-placement source into a caller-owned column-major matrix. */
export function composeRenderGlobalTransform(
    world: World,
    eid: number,
    out: Float32Array,
): Float32Array {
    const pose = world.resource(GlobalTransformHistory).scratch;
    readRenderGlobalTransform(world, eid, pose);
    return composeMat4(
        pose[0]!,
        pose[1]!,
        pose[2]!,
        pose[4]!,
        pose[5]!,
        pose[6]!,
        pose[7]!,
        pose[8]!,
        pose[9]!,
        pose[10]!,
        out,
    );
}
function stageHistoryDiscontinuities(world: World, runtime: GlobalTransformRuntime): Uint32Array {
    const marks = world.resource(TransformRuntime).discontinuities;
    if (runtime.historyDiscontinuitiesOnSubmit.length < marks.length) {
        const next = new Uint32Array(
            Math.max(marks.length, runtime.historyDiscontinuitiesOnSubmit.length * 2),
        );
        next.set(runtime.historyDiscontinuitiesOnSubmit);
        runtime.historyDiscontinuitiesOnSubmit = next;
    }
    const pending = runtime.historyDiscontinuitiesOnSubmit;
    for (let i = 0; i < pending.length; i++) {
        const combined = pending[i]! | (marks[i] ?? 0);
        pending[i] = combined;
        if (i < marks.length) marks[i] = 0;
    }
    return pending;
}
function markFrameWrites(world: World, runtime: GlobalTransformRuntime): void {
    const current = runtime.current!;
    const { global } = world.resource(TransformRuntime);
    const pd = world.fieldStorage(GlobalTransform, "translation").dirty;
    const qd = world.fieldStorage(GlobalTransform, "rotation").dirty;
    const sd = world.fieldStorage(GlobalTransform, "scale").dirty;
    const source = runtime.source;
    for (let word = 0; word < Math.max(pd.length, qd.length, sd.length); word++) {
        for (let bits = pd[word] | qd[word] | sd[word]; bits !== 0; bits &= bits - 1) {
            const eid = (word << 5) | (31 - Math.clz32(bits & -bits));
            const row = current.rowIndex(eid);
            if (row < 0) continue;
            const i = row * 12,
                j = eid * 4;
            const p = global.translation.column,
                q = global.rotation.column,
                s = global.scale.column;
            if (
                source[i] !== (p[j] ?? 0) ||
                source[i + 1] !== (p[j + 1] ?? 0) ||
                source[i + 2] !== (p[j + 2] ?? 0) ||
                source[i + 4] !== (q[j] ?? 0) ||
                source[i + 5] !== (q[j + 1] ?? 0) ||
                source[i + 6] !== (q[j + 2] ?? 0) ||
                source[i + 7] !== (q[j + 3] ?? 0) ||
                source[i + 8] !== (s[j] ?? 0) ||
                source[i + 9] !== (s[j + 1] ?? 0) ||
                source[i + 10] !== (s[j + 2] ?? 0)
            )
                resetInterpolation(world, eid);
        }
    }
}
function captureCurrent(world: World, phase: number): void {
    const runtime = world.resource(GlobalTransformHistory);
    const current = runtime.current!;
    runtime.cpuPrepared = false;
    current.prepareUpload();
    if (runtime.source.buffer !== current.bytes.buffer) {
        runtime.source = new Float32Array(
            current.bytes.buffer,
            current.bytes.byteOffset,
            current.bytes.byteLength / 4,
        );
    }
    // Reused stages retain earlier changed rows, including gaps in their range.
    if (current.pendingUploadSize && runtime.ranges[phase * 2 + 1])
        current.markRange(
            runtime.ranges[phase * 2] / current.rowBytes,
            runtime.ranges[phase * 2 + 1] / current.rowBytes,
        );
    const size = current.pendingUploadSize;
    let buffer = runtime.stages[phase];
    if (size && (!buffer || buffer.size < size)) {
        if (buffer) world.retireGpuResource(buffer);
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
function copyPhase(
    world: World,
    encoder: GPUCommandEncoder | undefined,
    phase: number,
    discontinuities?: Uint32Array,
): boolean {
    const runtime = world.resource(GlobalTransformHistory);
    const offset = runtime.ranges[phase * 2],
        size = runtime.ranges[phase * 2 + 1];
    if (size && !runtime.stages[phase])
        throw new Error(
            `GlobalTransform history phase ${phase} has ${size} bytes but no staging buffer`,
        );
    if (size)
        (encoder ?? world.frameEncoder()!).copyBufferToBuffer(
            runtime.stages[phase]!,
            0,
            runtime.current!.buffer,
            offset,
            size,
        );
    if (phase !== 1) return size > 0;
    if (runtime.historyDiscarded) {
        (encoder ?? world.frameEncoder()!).copyBufferToBuffer(
            runtime.current!.buffer,
            0,
            runtime.previous!.buffer,
            0,
            runtime.current!.buffer.size,
        );
        return size > 0;
    }
    const current = runtime.current!,
        marks = discontinuities ?? world.resource(TransformRuntime).discontinuities;
    for (let word = 0; word < marks.length; word++)
        for (let bits = marks[word]; bits !== 0; bits &= bits - 1) {
            const row = current.rowIndex((word << 5) | (31 - Math.clz32(bits & -bits)));
            if (row >= 0)
                (encoder ?? world.frameEncoder()!).copyBufferToBuffer(
                    current.buffer,
                    row * current.rowBytes,
                    runtime.previous!.buffer,
                    row * current.rowBytes,
                    current.rowBytes,
                );
        }
    return size > 0;
}
function discardJumpedHistory(world: World, runtime: GlobalTransformRuntime): void {
    if (runtime.fixedTickRevision === world.time.fixedTickRevision) return;
    runtime.fixedTickRevision = world.time.fixedTickRevision;
    recoverGlobalTransformHistory(world).restore();
    runtime.cpuPrepared = false;
    for (const history of runtime.cpuReaders) history.captured = false;
}
function prepareCpuFrame(world: World, runtime: GlobalTransformRuntime): void {
    discardJumpedHistory(world, runtime);
    if (runtime.cpuPrepared) return;
    runtime.cpuPrepared = true;
    const marks = world.resource(TransformRuntime).discontinuities;
    const pending = runtime.historyDiscontinuitiesOnSubmit;
    for (let i = 0; i < runtime.cpuReaders.length; i++) {
        const history = runtime.cpuReaders[i]!,
            eid = history.eid;
        const bit = 1 << (eid & 31),
            word = eid >>> 5;
        if (((marks[word] | pending[word]) & bit) !== 0) history.previous.set(history.current);
    }
}
function captureCpuHistory(world: World, runtime: GlobalTransformRuntime, start: boolean): void {
    runtime.cpuPrepared = false;
    const global = world.storage(GlobalTransform);
    const p = global.translation.column,
        q = global.rotation.column,
        s = global.scale.column;
    for (let i = 0; i < runtime.cpuReaders.length; i++) {
        const history = runtime.cpuReaders[i]!,
            j = history.eid * 4;
        const out = start ? history.previous : history.current;
        out[0] = p[j]!;
        out[1] = p[j + 1]!;
        out[2] = p[j + 2]!;
        out[4] = q[j]!;
        out[5] = q[j + 1]!;
        out[6] = q[j + 2]!;
        out[7] = q[j + 3]!;
        out[8] = s[j]!;
        out[9] = s[j + 1]!;
        out[10] = s[j + 2]!;
        if (start) {
            history.current.set(out);
            history.captured = true;
        }
    }
}
/** CPU rendered placement enrollment and fixed samples; requires no GPU. */
export const RenderTransformHistoryPlugin: Plugin = {
    name: "RenderTransformHistory",
    dependencies: [TransformPlugin],
    components: [RenderTransformHistory],
    systems: [GlobalTransformHistoryStartSystem, GlobalTransformHistoryEndSystem],
    recovery(world) {
        const runtime = world.resource(GlobalTransformHistory);
        return {
            snapshot: () => undefined,
            restore() {
                runtime.cpuPrepared = false;
                for (let i = 0; i < runtime.cpuReaders.length; i++)
                    runtime.cpuReaders[i]!.captured = false;
            },
        };
    },
};
/** @internal Records pending history work through the World's draw encoder. */
export function prepareGlobalTransformFrame(world: World, encoder?: GPUCommandEncoder): void {
    const runtime = world.resource(GlobalTransformHistory);
    if (!runtime.enabled) return;
    prepareCpuFrame(world, runtime);
    let historyNeedsPromotion = runtime.historyNeedsPromotion;
    const discontinuities =
        encoder === undefined
            ? stageHistoryDiscontinuities(world, runtime)
            : world.resource(TransformRuntime).discontinuities;
    if (runtime.tickCount) {
        const phase0Copied = copyPhase(world, encoder, 0);
        if (historyNeedsPromotion || phase0Copied) {
            (encoder ?? world.frameEncoder()!).copyBufferToBuffer(
                runtime.current!.buffer,
                0,
                runtime.previous!.buffer,
                0,
                runtime.current!.buffer.size,
            );
            historyNeedsPromotion = false;
        }
    }
    if (copyPhase(world, encoder, 1, discontinuities)) historyNeedsPromotion = true;
    runtime.historyPromotionOnSubmit = historyNeedsPromotion;
    if (encoder === undefined) {
        world.onFrameSubmit(runtime.commitOnSubmit);
    } else {
        runtime.tickCount = 0;
        runtime.cpuPrepared = false;
        runtime.ranges.fill(0);
        runtime.historyNeedsPromotion = runtime.historyPromotionOnSubmit;
        runtime.historyDiscarded = false;
        world.resource(TransformRuntime).discontinuities.fill(0);
    }
    uploadVisualCorrections(world, runtime.render!);
    if (!runtime.current!.count) return;
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
