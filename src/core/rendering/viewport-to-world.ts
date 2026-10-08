import { qRotate, type Ray, Viewports, type World } from "../../engine";
import { GlobalTransform } from "../transform";
import { Camera, CameraMode } from "./camera";
import { Views } from "./view";

/** A world ray from fixed-tick camera placement through CSS pixels relative to its bound canvas's
 * top-left. The direction is normalized; the origin is offset by `near` along the perspective ray.
 * Uses the canvas aspect, not a Resolution override. Returns null without camera placement or a
 * non-empty bound viewport. Pointer hover and crosshair coordinates belong to the caller. */
export function viewportToWorld(world: World, camera: number, x: number, y: number): Ray | null {
    if (camera < 0 || !world.has(camera, Camera) || !world.has(camera, GlobalTransform))
        return null;
    const view = world.resource(Views).get(camera);
    const viewport = view && world.resource(Viewports).get(view.viewportIndex);
    if (!viewport || viewport.cssWidth <= 0 || viewport.cssHeight <= 0) return null;
    const ndcX = (x / viewport.cssWidth) * 2 - 1;
    const ndcY = 1 - (y / viewport.cssHeight) * 2;
    const aspect = viewport.cssWidth / viewport.cssHeight;
    const params = world.storage(Camera);
    const global = world.storage(GlobalTransform);
    const qx = global.rotation.x.get(camera);
    const qy = global.rotation.y.get(camera);
    const qz = global.rotation.z.get(camera);
    const qw = global.rotation.w.get(camera);
    const near = params.near.get(camera);
    const orthographic = params.mode.get(camera) === CameraMode.Orthographic;
    if (!orthographic)
        return screenToRay(
            x,
            y,
            viewport.cssWidth,
            viewport.cssHeight,
            params.fov.get(camera),
            near,
            [
                global.translation.x.get(camera),
                global.translation.y.get(camera),
                global.translation.z.get(camera),
            ],
            [qx, qy, qz, qw],
        );
    const [dx, dy, dz] = qRotate(qx, qy, qz, qw, 0, 0, -1);
    const len = Math.hypot(dx, dy, dz) || 1;
    const dir: [number, number, number] = [dx / len, dy / len, dz / len];
    const offset = qRotate(
        qx,
        qy,
        qz,
        qw,
        (ndcX * aspect * params.size.get(camera)) / 2,
        (ndcY * params.size.get(camera)) / 2,
        -near,
    );
    return {
        origin: [
            global.translation.x.get(camera) + offset[0],
            global.translation.y.get(camera) + offset[1],
            global.translation.z.get(camera) + offset[2],
        ],
        dir,
    };
}

const DEG2RAD = Math.PI / 180;

/**
 * a world-space pick ray through a normalized-device-coordinate point (`ndcX`/`ndcY` in [-1, 1], x right /
 * y up; (0, 0) is screen centre). Unprojects through the camera's vertical `fov` (degrees) + `aspect`,
 * rotates the camera-space ray into world by the camera `quat`, and offsets the origin to the `near` plane.
 * The returned `dir` is normalized, so a hit distance is world units. Pair with {@link screenToRay}
 * for pixel input.
 */
export function generateRay(
    ndcX: number,
    ndcY: number,
    aspect: number,
    fov: number,
    near: number,
    origin: readonly [number, number, number],
    quat: readonly [number, number, number, number],
): Ray {
    const t = Math.tan((fov * DEG2RAD) / 2);
    const [dx, dy, dz] = qRotate(
        quat[0],
        quat[1],
        quat[2],
        quat[3],
        ndcX * aspect * t,
        ndcY * t,
        -1,
    );
    const len = Math.hypot(dx, dy, dz) || 1;
    const nx = dx / len;
    const ny = dy / len;
    const nz = dz / len;
    return {
        origin: [origin[0] + nx * near, origin[1] + ny * near, origin[2] + nz * near],
        dir: [nx, ny, nz],
    };
}

/**
 * a world-space pick ray through a canvas pixel (`screenX`/`screenY` in [0, width]×[0, height], origin
 * top-left). Converts the pixel to NDC + aspect and defers to {@link generateRay}: the cursor-driven pick
 * primitive (god-mode pick/drag). `origin`/`quat` are the camera's world pose, `fov`/`near` its params.
 */
export function screenToRay(
    screenX: number,
    screenY: number,
    width: number,
    height: number,
    fov: number,
    near: number,
    origin: readonly [number, number, number],
    quat: readonly [number, number, number, number],
): Ray {
    return generateRay(
        (screenX / width) * 2 - 1,
        1 - (screenY / height) * 2,
        width / height,
        fov,
        near,
        origin,
        quat,
    );
}
