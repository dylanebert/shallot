import { run, bench, group } from "mitata";
import {
    slerp,
    rotate,
    eulerToQuaternion,
    quaternionToEuler,
    lookAt,
} from "../src/engine/utils/math";
import { quat, mat4 } from "wgpu-matrix";

const ITERATIONS = 10_000;
const DEG_TO_RAD = Math.PI / 180;

export async function runMathBenchmarks() {
    console.log("\n=== Math Benchmarks (10k iterations) ===\n");

    const wgpuQuatA = quat.create();
    const wgpuQuatB = quat.create();
    const wgpuQuatOut = quat.create();
    const wgpuMat4 = mat4.create();
    const wgpuVec3A = new Float32Array(3);
    const wgpuVec3B = new Float32Array(3);
    const wgpuVec3C = new Float32Array(3);

    let accum = 0;

    group("eulerToQuaternion", () => {
        bench("shallot", () => {
            for (let i = 0; i < ITERATIONS; i++) {
                const r = eulerToQuaternion(i * 0.1, i * 0.2, i * 0.3);
                accum += r.x + r.y + r.z + r.w;
            }
        });

        bench("wgpu-matrix", () => {
            for (let i = 0; i < ITERATIONS; i++) {
                quat.fromEuler(
                    i * 0.1 * DEG_TO_RAD,
                    i * 0.2 * DEG_TO_RAD,
                    i * 0.3 * DEG_TO_RAD,
                    "xyz",
                    wgpuQuatOut,
                );
                accum += wgpuQuatOut[0] + wgpuQuatOut[1] + wgpuQuatOut[2] + wgpuQuatOut[3];
            }
        });
    });

    group("quaternionToEuler", () => {
        bench("shallot", () => {
            for (let i = 0; i < ITERATIONS; i++) {
                const t = i * 0.0001;
                const r = quaternionToEuler(t, t * 0.5, t * 0.3, Math.sqrt(1 - 3 * t * t));
                accum += r.x + r.y + r.z;
            }
        });

        bench("wgpu-matrix (no equivalent - inline same math)", () => {
            const out = { x: 0, y: 0, z: 0 };
            for (let i = 0; i < ITERATIONS; i++) {
                const t = i * 0.0001;
                const x = t,
                    y = t * 0.5,
                    z = t * 0.3,
                    w = Math.sqrt(1 - 3 * t * t);
                const x2 = x + x,
                    y2 = y + y,
                    z2 = z + z;
                const xx = x * x2,
                    xy = x * y2,
                    xz = x * z2;
                const yy = y * y2,
                    yz = y * z2;
                const wx = w * x2,
                    wy = w * y2,
                    wz = w * z2;
                const m13 = xz + wy;
                const ey = Math.asin(m13 < -1 ? -1 : m13 > 1 ? 1 : m13);
                if (m13 > -0.9999999 && m13 < 0.9999999) {
                    out.x = Math.atan2(wx - yz, 1 - (xx + yy)) * 57.29577951308232;
                    out.z = Math.atan2(wz - xy, 1 - (yy + z * z2)) * 57.29577951308232;
                } else {
                    out.x = Math.atan2(yz + wx, 1 - (xx + z * z2)) * 57.29577951308232;
                    out.z = 0;
                }
                out.y = ey * 57.29577951308232;
                accum += out.x + out.y + out.z;
            }
        });
    });

    group("rotate", () => {
        bench("shallot", () => {
            for (let i = 0; i < ITERATIONS; i++) {
                const r = rotate(0, 0, 0, 1, i * 0.01, i * 0.02, i * 0.03);
                accum += r.x + r.y + r.z + r.w;
            }
        });

        bench("wgpu-matrix", () => {
            for (let i = 0; i < ITERATIONS; i++) {
                quat.fromEuler(
                    i * 0.01 * DEG_TO_RAD,
                    i * 0.02 * DEG_TO_RAD,
                    i * 0.03 * DEG_TO_RAD,
                    "xyz",
                    wgpuQuatA,
                );
                quat.set(0, 0, 0, 1, wgpuQuatB);
                quat.multiply(wgpuQuatB, wgpuQuatA, wgpuQuatOut);
                accum += wgpuQuatOut[0] + wgpuQuatOut[1] + wgpuQuatOut[2] + wgpuQuatOut[3];
            }
        });
    });

    group("slerp", () => {
        bench("shallot", () => {
            for (let i = 0; i < ITERATIONS; i++) {
                const t = (i % 100) / 100;
                const r = slerp(0, 0, 0, 1, 0.5, 0.5, 0.5, 0.5, t);
                accum += r.x + r.y + r.z + r.w;
            }
        });

        bench("wgpu-matrix", () => {
            quat.set(0, 0, 0, 1, wgpuQuatA);
            quat.set(0.5, 0.5, 0.5, 0.5, wgpuQuatB);
            for (let i = 0; i < ITERATIONS; i++) {
                const t = (i % 100) / 100;
                quat.slerp(wgpuQuatA, wgpuQuatB, t, wgpuQuatOut);
                accum += wgpuQuatOut[0] + wgpuQuatOut[1] + wgpuQuatOut[2] + wgpuQuatOut[3];
            }
        });
    });

    group("lookAt", () => {
        bench("shallot", () => {
            for (let i = 0; i < ITERATIONS; i++) {
                const r = lookAt(i, i * 2, i * 3, 0, 0, 0);
                accum += r.x + r.y + r.z + r.w;
            }
        });

        bench("wgpu-matrix", () => {
            wgpuVec3B[0] = 0;
            wgpuVec3B[1] = 0;
            wgpuVec3B[2] = 0;
            wgpuVec3C[0] = 0;
            wgpuVec3C[1] = 1;
            wgpuVec3C[2] = 0;
            for (let i = 0; i < ITERATIONS; i++) {
                wgpuVec3A[0] = i;
                wgpuVec3A[1] = i * 2;
                wgpuVec3A[2] = i * 3;
                mat4.cameraAim(wgpuVec3A, wgpuVec3B, wgpuVec3C, wgpuMat4);
                quat.fromMat(wgpuMat4, wgpuQuatOut);
                accum += wgpuQuatOut[0] + wgpuQuatOut[1] + wgpuQuatOut[2] + wgpuQuatOut[3];
            }
        });
    });

    await run();

    if (accum === 0) console.log("prevent dead code elimination");
}

if (import.meta.main) {
    runMathBenchmarks();
}
