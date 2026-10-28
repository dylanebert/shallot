import type { World } from "../../engine";
import { CameraViewport } from "./camera";

export interface CameraRect {
    x: number;
    y: number;
    width: number;
    height: number;
}

/** Resolve coordinate-pixel bounds using Bevy's Viewport::clamp_to_size, including outside origins. */
export function cameraRect(
    world: World,
    eid: number,
    width: number,
    height: number,
    out: CameraRect,
): void {
    out.x = 0;
    out.y = 0;
    out.width = width;
    out.height = height;
    if (!world.has(eid, CameraViewport)) return;
    width = Math.floor(width);
    height = Math.floor(height);
    const rect = world.storage(CameraViewport);
    out.x = rect.x.get(eid);
    out.y = rect.y.get(eid);
    out.width = rect.width.get(eid);
    out.height = rect.height.get(eid);
    if (out.x + out.width > width) {
        if (out.x < width) out.width = width - out.x;
        else {
            out.x = Math.max(0, width - 1);
            out.width = width > 0 ? 1 : 0;
        }
    }
    if (out.y + out.height > height) {
        if (out.y < height) out.height = height - out.y;
        else {
            out.y = Math.max(0, height - 1);
            out.height = height > 0 ? 1 : 0;
        }
    }
}
