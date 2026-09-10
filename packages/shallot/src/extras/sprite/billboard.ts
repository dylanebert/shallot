// the executable spec for the sprite surface's billboard math (Godot material.cpp formulas — full
// screen-aligned = inverse-view columns substituted for the model rotation with scale re-applied;
// Y-locked = the cross(up, toViewer) orthonormal form), authored as pure TGSL fns so the same source
// the shader runs is callable here (testing.md's CPU-kernel tier — the draw itself is gated by the
// sprite gym scenario's framebuffer probe). `world` is the reconstructed instance matrix (`xformMat`);
// `right`/`up` are the camera's normalized world-space basis from the View uniform.

import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";

/** world-aligned: the plain instanced transform applied to the quad-local corner, no rotation
 *  substitution — `world * vec4(lx, ly, 0, 1)` collapsed to its xyz. */
export const worldCorner = tgpu
    .fn(
        [d.mat4x4f, d.f32, d.f32],
        d.vec3f,
    )((world, lx, ly) => {
        "use gpu";
        return std.add(
            std.add(std.mul(world.columns[0].xyz, lx), std.mul(world.columns[1].xyz, ly)),
            world.columns[3].xyz,
        );
    })
    .$name("worldCorner");

/** screen-aligned: camera right/up replace the model rotation; per-axis scale re-applied from the
 *  model columns' own length (rotation preserves length, so this recovers the authored scale without
 *  reading it separately). */
export const screenCorner = tgpu
    .fn(
        [d.mat4x4f, d.vec3f, d.vec3f, d.f32, d.f32],
        d.vec3f,
    )((world, right, up, lx, ly) => {
        "use gpu";
        const sx = lx * std.length(world.columns[0].xyz);
        const sy = ly * std.length(world.columns[1].xyz);
        return std.add(world.columns[3].xyz, std.add(std.mul(right, sx), std.mul(up, sy)));
    })
    .$name("screenCorner");

/**
 * Y-locked: the quad stays upright, yawing its horizontal axis toward the viewer. `cross(right, up)`
 * is the camera's backward axis (toward the viewer); its xz projection is the quad's facing. A
 * straight-down camera collapses that projection, so the camera up's xz projection (screen-up on the
 * ground plane) takes over; the quad is edge-on there anyway, the fallback only keeps the basis finite.
 */
export const yLockedCorner = tgpu
    .fn(
        [d.mat4x4f, d.vec3f, d.vec3f, d.f32, d.f32],
        d.vec3f,
    )((world, right, up, lx, ly) => {
        "use gpu";
        const toViewer = std.cross(right, up);
        let fx = toViewer.x;
        let fz = toViewer.z;
        if (fx * fx + fz * fz < 1e-8) {
            fx = up.x;
            fz = up.z;
        }
        const inv = 1 / std.length(d.vec2f(fx, fz));
        fx = fx * inv;
        fz = fz * inv;
        // quad right = cross((0,1,0), facing) — orthonormal with world up by construction
        const rx = fz;
        const rz = -fx;
        const sx = lx * std.length(world.columns[0].xyz);
        const sy = ly * std.length(world.columns[1].xyz);
        return d.vec3f(
            world.columns[3].x + rx * sx,
            world.columns[3].y + sy,
            world.columns[3].z + rz * sx,
        );
    })
    .$name("yLockedCorner");
