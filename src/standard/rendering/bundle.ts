import type { TgpuBindGroupLayout, TgpuRenderPipeline } from "typegpu";
import { isBuffer } from "typegpu";
import type { World } from "../../engine";
import type { Draw } from "./registry";

// Bundles avoid per-draw recording on frames whose draws, resource identities and pass shape are unchanged.

export interface BundleDraw {
    pipeline: TgpuRenderPipeline<never> | null;
    layout0: TgpuBindGroupLayout | null;
    group0: GPUBindGroup | null;
    layout1: TgpuBindGroupLayout | null;
    group1: GPUBindGroup | null;
    /** a backdrop's own group; null for a draw whose pipeline already carries its surface group */
    layout2: TgpuBindGroupLayout | null;
    group2: GPUBindGroup | null;
    /** the indirect args source, unwrapped only when the bundle is recorded; null for a vertex draw */
    indirect: Draw["args"]["indirect"] | GPUBuffer | null;
    /** byte offset into `indirect`, or the vertex count when `indirect` is null */
    offset: number;
}

export interface PassBundle {
    bundle: GPURenderBundle | null;
    /** the program as recorded, a capacity pool rewritten in place */
    program: BundleDraw[];
    count: number;
    /** the encoder shape the bundle was recorded against; a change in any of them re-records */
    sampleCount: number;
    colorFormat: GPUTextureFormat | null;
    depthFormat: GPUTextureFormat | null;
    /** the one-bundle list `executeBundles` replays, held so replaying mints nothing */
    replay: GPURenderBundle[];
}

/** an empty pass record; its program grows with the draw count and never shrinks */
export function newPassBundle(): PassBundle {
    return {
        bundle: null,
        program: [],
        count: 0,
        sampleCount: 0,
        colorFormat: null,
        depthFormat: null,
        replay: [],
    };
}

/** the draw record at `i`, appended at the high-water mark and rewritten in place below it */
export function bundleDraw(program: BundleDraw[], i: number): BundleDraw {
    let entry = program[i];
    if (entry) return entry;
    entry = {
        pipeline: null,
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
    descriptor: GPURenderBundleEncoderDescriptor,
): boolean {
    if (
        !pass.bundle ||
        pass.count !== count ||
        pass.sampleCount !== (descriptor.sampleCount ?? 1) ||
        pass.colorFormat !== (descriptor.colorFormats[0] ?? null) ||
        pass.depthFormat !== (descriptor.depthStencilFormat ?? null)
    ) {
        return true;
    }
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

/**
 * record `next` into `pass` as one render bundle against `descriptor`, and keep the program it was
 * recorded from. Called only at a transition {@link changed} reports.
 */
export function recordBundle(
    world: World,
    pass: PassBundle,
    next: readonly BundleDraw[],
    count: number,
    descriptor: GPURenderBundleEncoderDescriptor,
): void {
    const encoder = world.gpu.root["~unstable"].createRenderBundleEncoder(descriptor);
    for (let i = 0; i < count; i++) {
        const step = next[i];
        encoder.setPipeline(step.pipeline as TgpuRenderPipeline<never>);
        encoder.setBindGroup(step.layout0 as TgpuBindGroupLayout, step.group0 as GPUBindGroup);
        encoder.setBindGroup(step.layout1 as TgpuBindGroupLayout, step.group1 as GPUBindGroup);
        if (step.layout2 && step.group2) encoder.setBindGroup(step.layout2, step.group2);
        if (step.indirect) {
            const raw = isBuffer(step.indirect)
                ? world.gpu.root.unwrap(step.indirect)
                : (step.indirect as GPUBuffer);
            encoder.drawIndexedIndirect(raw, step.offset);
        } else {
            encoder.draw(step.offset);
        }
        const kept = bundleDraw(pass.program, i);
        kept.pipeline = step.pipeline;
        kept.layout0 = step.layout0;
        kept.group0 = step.group0;
        kept.layout1 = step.layout1;
        kept.group1 = step.group1;
        kept.layout2 = step.layout2;
        kept.group2 = step.group2;
        kept.indirect = step.indirect;
        kept.offset = step.offset;
    }
    pass.bundle = encoder.finish();
    pass.replay[0] = pass.bundle;
    pass.count = count;
    pass.sampleCount = descriptor.sampleCount ?? 1;
    pass.colorFormat = descriptor.colorFormats[0] ?? null;
    pass.depthFormat = descriptor.depthStencilFormat ?? null;
}
