import { qRotate, type Ray, Viewports, type World } from "../../engine";
import { GlobalTransform } from "../transform";
import { Camera, CameraMode } from "./camera";
import { cameraView } from "./view";

const _size = { width: 0, height: 0 };

// The camera's target size in its coordinate pixels: an attached texture's pixels, or the bound
// canvas's CSS pixels. Null for a depth-only view, an unbound camera or an empty target.
function targetSize(world: World, camera: number): { width: number; height: number } | null {
    const view = cameraView(world, camera);
    if (!view) return null;
    if (view.texture) {
        _size.width = view.texture.width;
        _size.height = view.texture.height;
    } else {
        const viewport = world.resource(Viewports).get(view.viewportIndex);
        if (!viewport) return null;
        _size.width = viewport.cssWidth;
        _size.height = viewport.cssHeight;
    }
    return _size.width > 0 && _size.height > 0 ? _size : null;
}

/** A world ray from fixed-tick camera placement through a pixel relative to the camera target's
 * top-left: CSS pixels of its bound canvas, or pixels of its attached texture. The direction is
 * normalized; the origin is offset by `near` along the perspective ray. Uses the target aspect, not a
 * Resolution override. Returns null without camera placement or a non-empty target. Pointer hover and
 * crosshair coordinates belong to the caller. */
export function viewportToWorld(world: World, camera: number, x: number, y: number): Ray | null {
    if (camera < 0 || !world.has(camera, Camera) || !world.has(camera, GlobalTransform))
        return null;
    const viewport = targetSize(world, camera);
    if (!viewport) return null;
    const width = viewport.width;
    const height = viewport.height;
    const ndcX = (x / width) * 2 - 1;
    const ndcY = 1 - (y / height) * 2;
    const aspect = width / height;
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
            width,
            height,
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

/** The pixel at which a world point appears, relative to the camera target's top-left, in the
 * coordinates {@link viewportToWorld} takes: CSS pixels of a bound canvas, or pixels of an attached
 * texture. Projects through fixed-tick camera placement with the target aspect, not a Resolution
 * override. Returns null without camera placement or a non-empty target, and for a point the camera
 * does not draw by depth: nearer than `near`, or, orthographic, beyond `far`. A point outside the
 * frame but in that depth range returns pixels outside the target. */
export function worldToViewport(
    world: World,
    camera: number,
    point: readonly [number, number, number],
): [number, number] | null {
    if (camera < 0 || !world.has(camera, Camera) || !world.has(camera, GlobalTransform))
        return null;
    const viewport = targetSize(world, camera);
    if (!viewport) return null;
    const params = world.storage(Camera);
    const global = world.storage(GlobalTransform);
    const [vx, vy, vz] = qRotate(
        -global.rotation.x.get(camera),
        -global.rotation.y.get(camera),
        -global.rotation.z.get(camera),
        global.rotation.w.get(camera),
        point[0] - global.translation.x.get(camera),
        point[1] - global.translation.y.get(camera),
        point[2] - global.translation.z.get(camera),
    );
    const depth = -vz;
    if (!(depth >= params.near.get(camera))) return null;
    const aspect = viewport.width / viewport.height;
    let ndcX: number;
    let ndcY: number;
    if (params.mode.get(camera) === CameraMode.Orthographic) {
        if (depth > params.far.get(camera)) return null;
        const size = params.size.get(camera);
        ndcX = vx / (size * aspect);
        ndcY = vy / size;
    } else {
        const t = Math.tan((params.fov.get(camera) * DEG2RAD) / 2);
        ndcX = vx / (depth * t * aspect);
        ndcY = vy / (depth * t);
    }
    return [((ndcX + 1) / 2) * viewport.width, ((1 - ndcY) / 2) * viewport.height];
}

const DEG2RAD = Math.PI / 180;

/**
 * World-space ray through NDC [-1,1] (x right, y up). fov is vertical degrees; quat is world rotation.
 * Origin lies on the near plane and dir is normalized, so hit distances are world units.
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

/** World-space pick ray through canvas pixels from top-left; origin/quat are world pose, fov is vertical degrees. */
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
