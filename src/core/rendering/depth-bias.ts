import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";

/** Biases reverse-Z clip depth toward the near plane exponentially or the far plane linearly; clamps bias to [-1, 1]. */
export const biasClipDepth = tgpu.fn(
    [d.vec4f, d.f32],
    d.vec4f,
)((clip, bias) => {
    "use gpu";
    const amount = std.clamp(bias, -1, 1);
    let depth = clip.z * (1 - amount);
    if (amount < 0) {
        // Bevy's epsilon keeps the -1 endpoint from rounding beyond the near plane.
        depth = clip.z * std.exp2(-amount * std.log2(clip.w / clip.z - 4.88e-4));
    }
    return d.vec4f(clip.xy, depth, clip.w);
});
