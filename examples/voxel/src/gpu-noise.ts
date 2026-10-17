import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { SEED } from "./terrain-data";

const hash = tgpu.fn(
    [d.i32, d.i32],
    d.u32,
)((x, z) => {
    "use gpu";
    let h = d.u32(x) * d.u32(374761393) + d.u32(z) * d.u32(668265263) + d.u32(SEED);
    h = (h ^ (h >>> 13)) * d.u32(1274126177);
    return h ^ (h >>> 16);
});
const gradient = tgpu.fn(
    [d.i32, d.i32, d.f32, d.f32],
    d.f32,
)((x, z, dx, dz) => {
    "use gpu";
    const h = hash(x, z) & 3;
    if (h === 0) return dx;
    if (h === 1) return -dx;
    if (h === 2) return dz;
    return -dz;
});
const fade = tgpu.fn(
    [d.f32],
    d.f32,
)((t) => {
    "use gpu";
    return t * t * t * (t * (t * 6 - 15) + 10);
});
export const perlin = tgpu.fn(
    [d.i32, d.i32],
    d.f32,
)((x, z) => {
    "use gpu";
    // Divide integers before converting to f32: fractional voxels survive beyond 2^24.
    const ix = x >> 5;
    const iz = z >> 5;
    const fx = d.f32(x & 31) / 32;
    const fz = d.f32(z & 31) / 32;
    return std.mix(
        std.mix(gradient(ix, iz, fx, fz), gradient(ix + 1, iz, fx - 1, fz), fade(fx)),
        std.mix(
            gradient(ix, iz + 1, fx, fz - 1),
            gradient(ix + 1, iz + 1, fx - 1, fz - 1),
            fade(fx),
        ),
        fade(fz),
    );
});
