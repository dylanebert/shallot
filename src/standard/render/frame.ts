import * as d from "typegpu/data";
import type { State } from "../../engine";
import { Compute } from "../../engine";
import { chunk, spliceNs } from "../../engine/utils/core";

/** the per-frame `Frame` UBO schema — the single source of truth for both sides of the layout (the
 * `View`/`Step` precedent): the emitted WGSL struct ({@link frameWgsl}) and the CPU staging write
 * ({@link writeFrame}, via `d.memoryLayoutOf`) both derive from it, so reordering a field can't leave one
 * side stamping the old offsets. Named `FrameGpu` (not `Frame`) because the CPU-side buffer + staging
 * singleton already owns that identifier ({@link Frame} below) — the `LightingGpu` precedent. */
export const FrameGpu = d
    .struct({
        time: d.f32,
        dt: d.f32,
        frame: d.u32,
    })
    .$name("Frame");

/** the Frame UBO's byte size: the schema's natural size (12 — three 4-byte scalars) rounded up to 16, the
 * alignment WGSL requires for any struct type bound in the uniform address space (`RequiredAlignOf` =
 * `max(AlignOf(S), 16)` — `d.sizeOf` alone under-reports it, since that rule is address-space-specific,
 * not a property of the struct itself). `View`/`Step`/`Lighting` all happen to land on a 16-byte multiple
 * already (their largest members force it); Frame is the first schema this port has hit that doesn't. */
export const FRAME_UNIFORM_SIZE = Math.ceil(d.sizeOf(FrameGpu) / 16) * 16;

const TIME_F32 = d.memoryLayoutOf(FrameGpu, (s) => s.time).offset / 4;
const DT_F32 = d.memoryLayoutOf(FrameGpu, (s) => s.dt).offset / 4;
const FRAME_U32 = d.memoryLayoutOf(FrameGpu, (s) => s.frame).offset / 4;

/** the per-frame `Frame` UBO's WGSL struct text, spliced by sear for every surface and by any
 * relocatable consumer that binds `frame`; emitted from {@link FrameGpu} under strict naming so the
 * struct text and the schema can never drift. */
export const frameWgsl = chunk("frameWgsl", [FrameGpu], spliceNs);

/**
 * GPU Frame UBO + CPU staging mirror, written once per frame by {@link writeFrame}
 * @expand
 */
export interface Frame {
    buffer: GPUBuffer;
    staging: Float32Array;
    stagingU32: Uint32Array;
}

const _backing = new ArrayBuffer(FRAME_UNIFORM_SIZE);

export const Frame: Frame = {
    buffer: null!,
    staging: new Float32Array(_backing),
    stagingU32: new Uint32Array(_backing),
};

/** pack time + frame counter into the Frame UBO */
export function writeFrame(state: State): void {
    if (!Compute.device || !Frame.buffer) return;
    Frame.staging[TIME_F32] = state.time.elapsed;
    Frame.staging[DT_F32] = state.time.deltaTime;
    Frame.stagingU32[FRAME_U32] = Compute.frame;
    Compute.device.queue.writeBuffer(Frame.buffer, 0, Frame.staging as Float32Array<ArrayBuffer>);
}
