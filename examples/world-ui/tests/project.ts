import { Camera, GlobalTransform, qRotate, type World } from "@dylanebert/shallot";

export function project(
    world: World,
    camera: number,
    position: readonly number[],
    width = 1280,
    height = 720,
): { x: number; y: number } {
    const transform = world.storage(GlobalTransform);
    const p = transform.translation;
    const q = transform.rotation;
    const [x, y, z] = qRotate(
        -q.x.get(camera),
        -q.y.get(camera),
        -q.z.get(camera),
        q.w.get(camera),
        position[0] - p.x.get(camera),
        position[1] - p.y.get(camera),
        position[2] - p.z.get(camera),
    );
    const tan = Math.tan((world.storage(Camera).fov.get(camera) * Math.PI) / 360);
    return {
        x: width * 0.5 + (x / (-z * tan)) * height * 0.5,
        y: height * 0.5 - (y / (-z * tan)) * height * 0.5,
    };
}
