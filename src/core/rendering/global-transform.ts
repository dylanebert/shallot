import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import type { GpuTable, Resource, System, World } from "../../engine/ecs";
import {
    GlobalTransform,
    GlobalTransformTickEndSystem,
    GlobalTransformTickStartSystem,
    PrepareGlobalTransformSystem,
    resetInterpolation,
    TransformRuntime,
} from "../transform";
import { PointLight, SpotLight } from "./lighting";
import { Xform } from "./xform";

type LightHistory = {
    eid: number;
    captured: boolean;
    stages: Float32Array[];
    current: Float32Array;
    previous: Float32Array;
};

const terms = [GlobalTransform];
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
/** @internal Rendering-owned GPU placement history. */
export interface GlobalTransformRuntime {
    current?: GpuTable<typeof Xform>;
    previous?: GpuTable<typeof Xform>;
    render?: GpuTable<typeof Xform>;
    enabled: boolean;
    tickCount: number;
    stages: (GPUBuffer | undefined)[];
    cpuPrepared: boolean;
    lights: LightHistory[];
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
    commitOnSubmit: () => void;
}
/** @internal GlobalTransform's GPU tick history and interpolated render table, with pending promotion and discontinuities. */
export const GlobalTransformHistory: Resource<GlobalTransformRuntime> = {
    create(world) {
        let runtime: GlobalTransformRuntime;
        runtime = {
            enabled: false,
            tickCount: 0,
            stages: new Array(2),
            cpuPrepared: false,
            lights: [],
            source: new Float32Array(0),
            ranges: new Uint32Array(4),
            generation: -1,
            historyNeedsPromotion: false,
            historyPromotionOnSubmit: false,
            historyDiscontinuitiesOnSubmit: new Uint32Array(1),
            historyDiscarded: false,
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
        world.observeMembership(GlobalTransform, (eid, present) => {
            if (present && runtime.enabled) resetInterpolation(world, eid);
        });
        const observeLight = (eid: number) => {
            if (!runtime.enabled) return;
            const index = runtime.lights.findIndex((light) => light.eid === eid);
            if (world.has(eid, PointLight) || world.has(eid, SpotLight)) {
                if (index < 0) addLightHistory(runtime, eid);
            } else if (index >= 0) {
                runtime.lights[index] = runtime.lights[runtime.lights.length - 1]!;
                runtime.lights.pop();
            }
        };
        world.observeMembership(PointLight, observeLight);
        world.observeMembership(SpotLight, observeLight);
        return runtime;
    },
};
function addLightHistory(runtime: GlobalTransformRuntime, eid: number): void {
    runtime.lights.push({
        eid,
        captured: false,
        stages: [new Float32Array(12), new Float32Array(12)],
        current: new Float32Array(12),
        previous: new Float32Array(12),
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
            if (world.has(eid, PointLight) || world.has(eid, SpotLight))
                addLightHistory(runtime, eid);
        }
    }
    return runtime.render!;
}
/** @internal Reads shadow placement for PointLight or SpotLight holders into a caller-owned
 * 12-float Xform record (position at 0, xyzw quaternion at 4, scale at 8).
 * Refuses entities holding neither PointLight nor SpotLight. Requires an enabled
 * globalTransformTable and the draw placement boundary. A newly attached or reattached
 * light reads current placement until the next fixed tick captures its history; thereafter
 * it matches drawn placement, using fixedAlpha, linear position/scale and shortest-path
 * normalized quaternion blend. No GPU readback is performed. */
export function readRenderGlobalTransform(world: World, eid: number, out: Float32Array): void {
    const runtime = world.resource(GlobalTransformHistory);
    if (!world.has(eid, PointLight) && !world.has(eid, SpotLight))
        throw new Error("readRenderGlobalTransform requires PointLight or SpotLight");
    prepareCpuFrame(world, runtime);
    const row = runtime.current?.rowIndex(eid) ?? -1;
    if (row < 0)
        throw new Error("readRenderGlobalTransform requires a captured GlobalTransform row");
    let light: LightHistory | undefined;
    for (let i = 0; i < runtime.lights.length; i++)
        if (runtime.lights[i]!.eid === eid) {
            light = runtime.lights[i];
            break;
        }
    if (!light) throw new Error("readRenderGlobalTransform requires captured light membership");
    if (!light.captured) {
        for (let i = 0; i < 12; i++) out[i] = runtime.source[row * 12 + i]!;
        return;
    }
    const offset = 0;
    const current = light.current,
        previous = light.previous;
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
    const start = runtime.ranges[phase * 2] / 4;
    const end = start + runtime.ranges[phase * 2 + 1] / 4;
    for (let index = 0; index < runtime.lights.length; index++) {
        const light = runtime.lights[index]!;
        const row = current.rowIndex(light.eid);
        if (row < 0) continue;
        const offset = row * 12;
        if (phase === 0 && !light.captured) {
            for (let i = 0; i < 12; i++) {
                const value = runtime.source[offset + i]!;
                light.current[i] = light.previous[i] = value;
                light.stages[0]![i] = light.stages[1]![i] = value;
            }
            light.captured = true;
        }
        if (size && offset >= start && offset < end)
            for (let i = 0; i < 12; i++) light.stages[phase]![i] = runtime.source[offset + i]!;
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
function prepareCpuFrame(world: World, runtime: GlobalTransformRuntime): void {
    if (runtime.cpuPrepared || !runtime.enabled) return;
    runtime.cpuPrepared = true;
    const marks = world.resource(TransformRuntime).discontinuities;
    const pending = runtime.historyDiscontinuitiesOnSubmit;
    for (let index = 0; index < runtime.lights.length; index++) {
        const light = runtime.lights[index]!;
        if (!light.captured) continue;
        const offset = runtime.current!.rowIndex(light.eid) * 12;
        if (offset < 0) continue;
        for (let phase = 0; phase < 2; phase++) {
            if (phase === 0 && !runtime.tickCount) continue;
            const start = runtime.ranges[phase * 2] / 4;
            const size = runtime.ranges[phase * 2 + 1] / 4;
            if (offset >= start && offset < start + size) light.current.set(light.stages[phase]!);
            if (
                (phase === 0 && (runtime.historyNeedsPromotion || size > 0)) ||
                (phase === 1 && runtime.historyDiscarded)
            )
                light.previous.set(light.current);
        }
        const word = light.eid >>> 5;
        if (((marks[word] | pending[word]) & (1 << (light.eid & 31))) !== 0)
            light.previous.set(light.current);
    }
}
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
