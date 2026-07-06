import type { State } from "../../engine";
import { Compute } from "../../engine";

export const FRAME_UNIFORM_SIZE = 16;

const TIME_F32 = 0;
const DT_F32 = 1;
const FRAME_U32 = 2;

/** the per-frame `Frame` UBO's WGSL struct (time / dt / frame counter) — spliced by sear for every
 * surface and by any relocatable consumer that binds `frame`; layout mirrors {@link Frame}. */
export const FRAME_STRUCT_WGSL = /* wgsl */ `
struct Frame {
    time: f32,
    dt: f32,
    frame: u32,
}`;

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
