import * as d from "typegpu/data";
import type { World } from "../../engine";

import { chunk, spliceNs } from "../../engine/utils";

/** the per-frame `Frame` UBO schema — the single source of truth for both sides of the layout (the
 * `View`/`Step` precedent): the emitted WGSL struct ({@link frameWgsl}) and the CPU staging write
 * ({@link writeFrame}, via `d.memoryLayoutOf`) both derive from it, so reordering a field can't leave one
 * side stamping the old offsets. Named `FrameGpu` (not `Frame`) because the CPU-side buffer + staging
 * singleton already owns that identifier ({@link Frame} below) — the `LightingGpu` precedent. */
export const FrameGpu = d
    .struct({
        globalTransformParams: d.vec4f,
        time: d.f32,
        dt: d.f32,
        frame: d.u32,
    })
    .$name("Frame");

/** the Frame UBO's byte size: its 28-byte schema rounded up to 32, the alignment WGSL requires for any
 * struct type bound in the uniform address space (`RequiredAlignOf` = `max(AlignOf(S), 16)`). The first
 * vec4 holds GlobalTransform interpolation parameters; `d.sizeOf` excludes only the final 4-byte pad. */
export const FRAME_UNIFORM_SIZE = Math.ceil(d.sizeOf(FrameGpu) / 16) * 16;

const GLOBAL_TRANSFORM_PARAMS_F32 =
    d.memoryLayoutOf(FrameGpu, (s) => s.globalTransformParams).offset / 4;
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

export const frameKey = { create: createFrame };

function createFrame(): Frame {
    const backing = new ArrayBuffer(FRAME_UNIFORM_SIZE);
    return {
        buffer: null!,
        staging: new Float32Array(backing),
        stagingU32: new Uint32Array(backing),
    };
}

/** Create this world's frame UBO state during RenderingPlugin initialization. */
export function initializeFrameState(world: World): void {
    world.resource(frameKey);
}

export const Frame: import("../../engine").Resource<Frame> = {
    create: (world) => world.resource(frameKey),
};

/** Pack interpolation parameters, time, and frame counter into the shared Frame UBO. */
export function writeFrame(world: World): void {
    const _frame = world.resource(Frame);

    if (!world.gpu.device || !_frame.buffer) return;
    const globalTransform = world.globalTransformRuntime;
    if (globalTransform?.enabled) {
        _frame.staging[GLOBAL_TRANSFORM_PARAMS_F32] = world.time.fixedAlpha;
        _frame.staging[GLOBAL_TRANSFORM_PARAMS_F32 + 1] = globalTransform.current?.count ?? 0;
    }
    _frame.staging[TIME_F32] = world.time.elapsed;
    _frame.staging[DT_F32] = world.time.deltaTime;
    _frame.stagingU32[FRAME_U32] = world.gpu.frame;
    world.gpu.device.queue.writeBuffer(
        _frame.buffer,
        0,
        _frame.staging as Float32Array<ArrayBuffer>,
    );
}
