import type { TgpuRoot } from "typegpu";
import { Compute } from "../../engine";

/**
 * reject a device the builder can't build pipelines on. Its typed pipelines and bind groups come from
 * `Compute.root`, which is scoped to `Compute.device` (memoized there, `engine/runtime/gpu.ts`), while a
 * stage factory takes the device its buffers are allocated on — so a mismatch is a cross-device
 * dispatch, a validation error with nothing pointing back here. Every entry point calls this (or {@link
 * rootOf}) **before its first allocation** — the throw has to land while there is nothing to free.
 * Narrowed exactly like the image-array blit path (`render/image.ts`); one root per device lands at 4a
 * and deletes both guards.
 * @internal
 */
export function checkDevice(device: GPUDevice, stage: string): void {
    if (device !== Compute.device)
        throw new Error(
            `[bvh] ${stage} builds its pipelines on Compute.device — pass that device, or call ` +
                "requestGPU with the one you want the builder to use",
        );
}

/** {@link checkDevice}, returning the root the caller then builds its pipelines through.
 *  @internal */
export function rootOf(device: GPUDevice, stage: string): TgpuRoot {
    checkDevice(device, stage);
    return Compute.root;
}
