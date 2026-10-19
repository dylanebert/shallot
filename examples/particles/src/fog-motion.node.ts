import { expect, setDefaultTimeout, test } from "bun:test";
import { Transform } from "@dylanebert/shallot";
import { NoFog } from "@dylanebert/shallot/extras";
import { attachTexture, captureTexture } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { Motes, MotesPlugin } from "./motes";
import { Player } from "./player";
import { Room } from "./room";

setDefaultTimeout(TIMEOUT.node);
const WIDTH = 640;
const HEIGHT = 360;
// The room camera's vertical field of view.
const FOV = 70;
function linear(value: number) {
    const x = value / 255;
    return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
}
type Vec3 = readonly [number, number, number];
type Pose = { eye: Vec3; yaw: number; pitch: number };
/** Yaw about +y, then pitch about the camera's x. */
function rotation({ yaw, pitch }: Pose): [number, number, number, number] {
    const cy = Math.cos(yaw / 2);
    const sy = Math.sin(yaw / 2);
    const cp = Math.cos(pitch / 2);
    const sp = Math.sin(pitch / 2);
    return [cy * sp, sy * cp, -sy * sp, cy * cp];
}
/** A world point's pixel, or undefined off screen or behind the camera. */
function project(pose: Pose, point: Vec3): readonly [number, number] | undefined {
    const [qx, qy, qz, qw] = rotation(pose);
    const v = [point[0] - pose.eye[0], point[1] - pose.eye[1], point[2] - pose.eye[2]];
    // Rotate into the camera's frame by the inverse rotation.
    const [x, y, z] = [-qx, -qy, -qz];
    const tx = 2 * (y * v[2]! - z * v[1]!);
    const ty = 2 * (z * v[0]! - x * v[2]!);
    const tz = 2 * (x * v[1]! - y * v[0]!);
    const cx = v[0]! + qw * tx + (y * tz - z * ty);
    const cy = v[1]! + qw * ty + (z * tx - x * tz);
    const cz = v[2]! + qw * tz + (x * ty - y * tx);
    if (cz >= 0) return undefined;
    const tangent = Math.tan((FOV * Math.PI) / 360);
    const px = Math.round(((cx / -cz / (tangent * (WIDTH / HEIGHT)) + 1) * WIDTH) / 2);
    const py = Math.round(((1 - cy / -cz / tangent) * HEIGHT) / 2);
    if (px < 2 || py < 2 || px >= WIDTH - 2 || py >= HEIGHT - 2) return undefined;
    return [px, py];
}
function patch(rgba: Uint8ClampedArray, [cx, cy]: readonly [number, number]) {
    let sum = 0;
    for (let y = cy - 2; y <= cy + 2; y++) {
        for (let x = cx - 2; x <= cx + 2; x++) {
            const at = (y * WIDTH + x) * 4;
            sum +=
                0.2126 * linear(rgba[at]!) +
                0.7152 * linear(rgba[at + 1]!) +
                0.0722 * linear(rgba[at + 2]!);
        }
    }
    return sum / 25;
}
// World-fixed points on the room's inner seams and side walls, away from the sun shaft, where light from the
// sunlit air outside the walls would show.
const SEAMS: readonly Vec3[] = [
    [-3.3, 3.2, -2],
    [3.3, 3.2, -2],
    [-3.3, 2, -3.3],
    [3.3, 2, -3.3],
    [-1.6, 3.2, -3.3],
    [-3.3, 1.5, -1],
    [3.3, 1.5, -1],
];
// The back-right corner, where one column holds the back wall's surface beside the side wall's.
const CORNER: readonly Vec3[] = [1, 1.4, 1.8, 2.2, 2.6].map((y) => [3.3, y, -3.3] as const);
const FRAMES = 32;
const SAMPLED = [16, 32];

/**
 * Move the camera at the player's walk speed for 32 frames from a converged still start, then compare the
 * sampled frames' seam patches with the converged still reference at each pose: the equal average of all 16
 * phases, each from fresh history.
 */
async function seamError(
    path: (speed: number, frame: number) => Pose,
    points: readonly Vec3[] = SEAMS,
) {
    const app = await openGpuApp({ plugins: [MotesPlugin] });
    try {
        const { world } = app;
        const { camera, player } = world.resource(Room);
        const speed = world.storage(Player).speed.get(player);
        world.remove(player, Player);
        world.resource(Motes).visible = false;
        attachTexture(world, camera, { width: WIDTH, height: HEIGHT });
        const place = (frame: number) => {
            const pose = path(speed, frame);
            world.storage(Transform).translation.set(camera, ...pose.eye, 0);
            world.storage(Transform).rotation.set(camera, ...rotation(pose));
            world.tick();
            world.step(0);
        };
        for (let i = 0; i < 64; i++) {
            place(0);
            await world.gpu.sync();
        }
        const moving = new Map<number, Uint8ClampedArray>();
        for (let frame = 1; frame <= FRAMES; frame++) {
            place(frame);
            if (SAMPLED.includes(frame))
                moving.set(frame, (await captureTexture(world, camera)).rgba.slice());
        }
        let error = 0;
        let seen = 0;
        for (const [frame, rgba] of moving) {
            const pixels = points.map((point) => project(path(speed, frame), point));
            const reference = points.map(() => 0);
            for (let i = 0; i < 16; i++) {
                // Opt-out discards history. A three-frame stride visits all 16 Halton phases.
                world.add(camera, NoFog);
                place(frame);
                place(frame);
                world.remove(camera, NoFog);
                place(frame);
                const still = (await captureTexture(world, camera)).rgba;
                pixels.forEach((pixel, k) => {
                    if (pixel) reference[k]! += patch(still, pixel) / 16;
                });
            }
            pixels.forEach((pixel, k) => {
                if (!pixel) return;
                seen++;
                error = Math.max(error, Math.abs(patch(rgba, pixel) - reference[k]!));
            });
        }
        console.log("seam motion error", { error, seen });
        return { error, seen };
    } finally {
        app.dispose();
    }
}

// fog.node.ts holds the still camera's shaft edge to 0.005; a moving camera's seams are held to the same.
test("seams hold the still reference while walking forward", async () => {
    const { error, seen } = await seamError((speed, frame) => ({
        eye: [0.4, 1.7, 2.6 - (speed / 60) * frame],
        yaw: 0,
        pitch: 0.25,
    }));
    expect(seen).toBeGreaterThan(SEAMS.length);
    expect(error).toBeLessThan(0.005);
});

test("seams hold the still reference while strafing and turning", async () => {
    const { error, seen } = await seamError((speed, frame) => ({
        eye: [-1.4 + (speed / 60) * frame, 1.7, 2],
        yaw: -0.4 + (1.5 / 60) * frame,
        pitch: 0.25,
    }));
    expect(seen).toBeGreaterThan(SEAMS.length);
    expect(error).toBeLessThan(0.005);
});

test("the corner seam holds the still reference while dollying toward it and back", async () => {
    const { error, seen } = await seamError(
        (speed, frame) => ({
            eye: [3, 1.7, 3 - (speed / 60) * Math.min(frame, FRAMES - frame)],
            yaw: -0.02,
            pitch: -0.1,
        }),
        CORNER,
    );
    expect(seen).toBe(2 * CORNER.length);
    expect(error).toBeLessThan(0.005);
});
