import type { TgpuBindGroupLayout, TgpuRenderPipeline } from "typegpu";
import { isBuffer } from "typegpu";
import type { World } from "../../engine";
import {
    draw,
    drawIndexedIndirect,
    setIndexBuffer,
    setRenderBindGroup,
    setRenderPipeline,
} from "../../engine/runtime";
import { rethrowMaterialError } from "./material-type";
import type { Draw } from "./registry";

// A pass program is recorded once as the raw commands TypeGPU emits for it and replayed into the pass each
// frame, re-recorded only when a draw or resource identity changes. It is not a GPURenderBundle: WebKit
// replays a bundle holding an indirect draw into fresh Metal indirect command buffers on every
// executeBundles (https://bugs.webkit.org/show_bug.cgi?id=320490), which on macOS 26.7 held back each such
// pass's completion about 16 ms and capped a 120 Hz display at 60 fps.

export interface BundleDraw {
    pipeline: TgpuRenderPipeline<never> | null;
    material: string | null;
    layout0: TgpuBindGroupLayout | null;
    group0: GPUBindGroup | null;
    layout1: TgpuBindGroupLayout | null;
    group1: GPUBindGroup | null;
    /** a backdrop's own group; null for a draw whose pipeline already carries its surface group */
    layout2: TgpuBindGroupLayout | null;
    group2: GPUBindGroup | null;
    /** the indirect args source, unwrapped only when the program is recorded; null for a vertex draw */
    indirect: Draw["args"]["indirect"] | GPUBuffer | null;
    /** byte offset into `indirect`, or the vertex count when `indirect` is null */
    offset: number;
}

const SET_PIPELINE = 0;
const SET_BIND_GROUP = 1;
const SET_INDEX_BUFFER = 2;
const DRAW_INDEXED_INDIRECT = 3;
const DRAW = 4;

export interface PassBundle {
    recorded: boolean;
    /** the program as recorded, a capacity pool rewritten in place */
    program: BundleDraw[];
    count: number;
    /** recorded pass commands: one opcode each, with two operand slots per opcode */
    ops: number[];
    operands: unknown[];
}

/** an empty pass record; its program grows with the draw count and never shrinks */
export function newPassBundle(): PassBundle {
    return { recorded: false, program: [], count: 0, ops: [], operands: [] };
}

/** the draw record at `i`, appended at the high-water mark and rewritten in place below it */
export function bundleDraw(program: BundleDraw[], i: number): BundleDraw {
    let entry = program[i];
    if (entry) return entry;
    entry = {
        pipeline: null,
        material: null,
        layout0: null,
        group0: null,
        layout1: null,
        group1: null,
        layout2: null,
        group2: null,
        indirect: null,
        offset: 0,
    };
    program[i] = entry;
    return entry;
}

export function bundleChanged(
    pass: PassBundle,
    next: readonly BundleDraw[],
    count: number,
): boolean {
    if (!pass.recorded || pass.count !== count) return true;
    for (let i = 0; i < count; i++) {
        const a = pass.program[i];
        const b = next[i];
        if (
            a.pipeline !== b.pipeline ||
            a.layout0 !== b.layout0 ||
            a.group0 !== b.group0 ||
            a.layout1 !== b.layout1 ||
            a.group1 !== b.group1 ||
            a.layout2 !== b.layout2 ||
            a.group2 !== b.group2 ||
            a.indirect !== b.indirect ||
            a.offset !== b.offset
        ) {
            return true;
        }
    }
    return false;
}

function emit(pass: PassBundle, op: number, a: unknown, b?: unknown): void {
    pass.ops.push(op);
    pass.operands.push(a, b);
}

// A render bundle encoder by shape, so TypeGPU emits into it what it would into a real one. Only the
// commands Standard's draws emit are present; any other is a TypeError at record time.
function recorder(pass: PassBundle) {
    return {
        setPipeline: (pipeline: GPURenderPipeline) => emit(pass, SET_PIPELINE, pipeline),
        setBindGroup: (index: number, group: GPUBindGroup, offsets?: unknown) => {
            if (offsets !== undefined) throw new Error("pass program: dynamic offsets unsupported");
            emit(pass, SET_BIND_GROUP, index, group);
        },
        setIndexBuffer: (
            buffer: GPUBuffer,
            format: GPUIndexFormat,
            offset?: number,
            size?: number,
        ) => {
            if (offset !== undefined || size !== undefined)
                throw new Error("pass program: index buffer ranges unsupported");
            emit(pass, SET_INDEX_BUFFER, buffer, format);
        },
        drawIndexedIndirect: (buffer: GPUBuffer, offset: number) =>
            emit(pass, DRAW_INDEXED_INDIRECT, buffer, offset),
        draw: (vertexCount: number, ...rest: unknown[]) => {
            if (rest.some((value) => value !== undefined))
                throw new Error("pass program: only a vertex count is supported");
            emit(pass, DRAW, vertexCount);
        },
        finish: () => undefined,
    } as unknown as GPURenderBundleEncoder;
}

/**
 * record `next` into `pass` as the commands TypeGPU emits for it, and keep the program it was recorded
 * from. Called only at a transition {@link bundleChanged} reports.
 */
export function recordBundle(
    world: World,
    pass: PassBundle,
    next: readonly BundleDraw[],
    count: number,
): void {
    pass.ops.length = 0;
    pass.operands.length = 0;
    const into = recorder(pass);
    for (let i = 0; i < count; i++) {
        const step = next[i];
        let pipeline = (step.pipeline as TgpuRenderPipeline<never>)
            .with(step.layout0 as TgpuBindGroupLayout, step.group0 as GPUBindGroup)
            .with(step.layout1 as TgpuBindGroupLayout, step.group1 as GPUBindGroup);
        if (step.layout2 && step.group2) pipeline = pipeline.with(step.layout2, step.group2);
        pipeline = pipeline.with(into);
        try {
            if (step.indirect) {
                const raw = isBuffer(step.indirect)
                    ? world.gpu.root.unwrap(step.indirect)
                    : (step.indirect as GPUBuffer);
                pipeline.drawIndexedIndirect(raw, step.offset);
            } else {
                pipeline.draw(step.offset);
            }
        } catch (error) {
            if (step.material) rethrowMaterialError(step.material, error);
            throw error;
        }
        const kept = bundleDraw(pass.program, i);
        kept.pipeline = step.pipeline;
        kept.material = step.material;
        kept.layout0 = step.layout0;
        kept.group0 = step.group0;
        kept.layout1 = step.layout1;
        kept.group1 = step.group1;
        kept.layout2 = step.layout2;
        kept.group2 = step.group2;
        kept.indirect = step.indirect;
        kept.offset = step.offset;
    }
    pass.count = count;
    pass.recorded = true;
}

/** replay the recorded commands into `encoder`; a pass never recorded encodes nothing */
export function replayBundle(encoder: GPURenderPassEncoder, pass: PassBundle): void {
    const { ops, operands } = pass;
    for (let i = 0; i < ops.length; i++) {
        const a = operands[2 * i];
        const b = operands[2 * i + 1];
        switch (ops[i]) {
            case SET_PIPELINE:
                setRenderPipeline(encoder, a as GPURenderPipeline);
                break;
            case SET_BIND_GROUP:
                setRenderBindGroup(encoder, a as number, b as GPUBindGroup);
                break;
            case SET_INDEX_BUFFER:
                setIndexBuffer(encoder, a as GPUBuffer, b as GPUIndexFormat);
                break;
            case DRAW_INDEXED_INDIRECT:
                drawIndexedIndirect(encoder, a as GPUBuffer, b as number);
                break;
            default:
                draw(encoder, a as number);
        }
    }
}
