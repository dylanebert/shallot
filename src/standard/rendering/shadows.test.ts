import { expect, test } from "bun:test";
import { CameraMode } from "../../core/rendering";
import { lookAt, multiplyMat4, orthographic } from "../../engine";
import { cascadeFit, MAX_POINT_CASTERS, type PointShadowFrame, packCasters } from "./shadows";

// Reuse sparse rect storage across sets, as updatePointShadows does.
test("packCasters fills an equal grid for spots, points and mixed sets, dropping when MIN_TILE cannot fit", () => {
    const rects = new Float64Array(MAX_POINT_CASTERS * 6 * 4);
    const cases = [
        {
            name: "spots",
            side: 1024,
            spots: [true, true, true],
            count: 3,
            tiles: [[[0, 0, 512]], [[512, 0, 512]], [[0, 512, 512]]],
        },
        {
            name: "point",
            side: 1024,
            spots: [false],
            count: 1,
            tiles: [
                [
                    [0, 0, 256],
                    [256, 0, 256],
                    [512, 0, 256],
                    [768, 0, 256],
                    [0, 256, 256],
                    [256, 256, 256],
                ],
            ],
        },
        {
            name: "mixed",
            side: 1024,
            spots: [true, false, true],
            count: 3,
            tiles: [
                [[0, 0, 256]],
                [
                    [256, 0, 256],
                    [512, 0, 256],
                    [768, 0, 256],
                    [0, 256, 256],
                    [256, 256, 256],
                    [512, 256, 256],
                ],
                [[768, 256, 256]],
            ],
        },
        {
            name: "drop point at minimum",
            side: 128,
            spots: [true, false],
            count: 1,
            tiles: [[[0, 0, 128]]],
        },
        { name: "atlas below minimum", side: 32, spots: [true], count: 0, tiles: [] },
    ];
    for (const c of cases) {
        const frames = c.spots.map((spot) => ({ spot }) as PointShadowFrame);
        let count = frames.length;
        while (count > 0 && !packCasters(frames, count, c.side, rects)) count--;
        const tiles = frames.slice(0, count).map((frame, i) =>
            Array.from({ length: frame.spot ? 1 : 6 }, (_, face) => {
                const at = (i * 6 + face) * 4;
                expect(rects[at + 3]).toBe(rects[at + 2]);
                return [rects[at] * c.side, rects[at + 1] * c.side, rects[at + 2] * c.side];
            }),
        );
        expect({ name: c.name, count, tiles }).toEqual({
            name: c.name,
            count: c.count,
            tiles: c.tiles,
        });
    }
});

// The receiver reads a fragment outside its cascade box as lit, so a slice corner the box misses shows as a
// lighter wedge of shadow at the screen edge. Texel snapping moves the box by at most half a texel.
test("cascadeFit's light box holds every corner of its camera slice, at any lens, camera and sun angle", () => {
    const resolution = 2048;
    const fit = {
        eye: new Float64Array(3),
        focus: new Float64Array(3),
        up: new Float64Array(3),
        extent: new Float64Array(2),
    };
    const cam = new Float32Array(16);
    const proj = new Float32Array(16);
    const view = new Float32Array(16);
    const vp = new Float32Array(16);
    const slices = [
        [0.1, 6],
        [6, 20],
        [57, 150],
    ];
    const suns = [
        [0.3, -1, 0.2],
        [1, -0.4, 0],
        [-0.5, -0.7, 0.8],
        [0, -1, 0],
    ];
    let worstXY = 0;
    let worstZ = 0;
    for (const fov of [20, 60, 100]) {
        const tanH = Math.tan((fov * Math.PI) / 360);
        for (const aspect of [9 / 16, 16 / 9, 21 / 9]) {
            for (let yaw = 0; yaw < 360; yaw += 30) {
                for (let pitch = -80; pitch <= 80; pitch += 40) {
                    const cy = Math.cos((yaw * Math.PI) / 180);
                    const sy = Math.sin((yaw * Math.PI) / 180);
                    const cp = Math.cos((pitch * Math.PI) / 180);
                    const sp = Math.sin((pitch * Math.PI) / 180);
                    const f = [-sy * cp, sp, -cy * cp];
                    const r = [cy, 0, -sy];
                    const u = [
                        r[1] * f[2] - r[2] * f[1],
                        r[2] * f[0] - r[0] * f[2],
                        r[0] * f[1] - r[1] * f[0],
                    ];
                    cam.set([...r, 0, ...u, 0, -f[0], -f[1], -f[2], 0, 3, 2, -5, 1]);
                    for (const sun of suns) {
                        for (const [near, far] of slices) {
                            cascadeFit(
                                cam,
                                CameraMode.Perspective,
                                fov,
                                1,
                                aspect,
                                near,
                                far,
                                sun,
                                resolution,
                                40,
                                fit,
                            );
                            // the receiver's viewProj, as updateCascades composes it
                            orthographic(fit.extent[0], 1, 0, fit.extent[1], proj);
                            const { eye, focus, up } = fit;
                            lookAt(
                                eye[0],
                                eye[1],
                                eye[2],
                                focus[0],
                                focus[1],
                                focus[2],
                                up[0],
                                up[1],
                                up[2],
                                view,
                            );
                            multiplyMat4(proj, view, vp);
                            for (let corner = 0; corner < 8; corner++) {
                                const z = corner & 4 ? far : near;
                                const h = z * tanH * (corner & 2 ? 1 : -1);
                                const w = z * tanH * aspect * (corner & 1 ? 1 : -1);
                                const p = [0, 1, 2].map(
                                    (k) => cam[12 + k] + f[k] * z + r[k] * w + u[k] * h,
                                );
                                const x = vp[0] * p[0] + vp[4] * p[1] + vp[8] * p[2] + vp[12];
                                const y = vp[1] * p[0] + vp[5] * p[1] + vp[9] * p[2] + vp[13];
                                const depth = vp[2] * p[0] + vp[6] * p[1] + vp[10] * p[2] + vp[14];
                                worstXY = Math.max(worstXY, Math.abs(x), Math.abs(y));
                                worstZ = Math.max(worstZ, Math.abs(depth * 2 - 1));
                            }
                        }
                    }
                }
            }
        }
    }
    expect(worstXY).toBeLessThanOrEqual(1 + 1 / resolution);
    expect(worstZ).toBeLessThanOrEqual(1 + 1e-6);
});
