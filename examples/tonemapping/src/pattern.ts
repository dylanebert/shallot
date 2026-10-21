import { MaterialPlugin, materialType } from "@dylanebert/shallot/standard/rendering";
import * as d from "typegpu/data";
import * as std from "typegpu/std";

// Bevy's tonemapping_test_patterns.wesl; presentation applies the selected curve.
export const PatternMaterial = materialType({
    name: "TonemappingTestPatterns",
    parameters: d.struct({ unused: d.f32 }),
    defaults: { unused: 0 },
    fragmentInputs: { uv: true },
    fragment: (ctx) => {
        "use gpu";
        const uv = ctx.uv;
        if (uv.y > 0.5) {
            const y = (1 - uv.y) * 2 * (1 + 1 / 24);
            const h = (Math.PI * 2 * std.floor(1 + 24 * y)) / 24;
            const value = std.exp(15 * (std.floor(uv.x * 48) / 48 - 0.5));
            if (y >= 1) return d.vec4f(d.vec3f(value), 1);
            const color = std.cos(
                std.add(d.vec3f(h), std.mul(d.vec3f(0, 1, 2), (Math.PI * 2) / 3)),
            );
            const lo = std.min(color.x, std.min(color.y, color.z));
            const hi = std.max(color.x, std.max(color.y, color.z));
            return d.vec4f(std.mul(std.sub(color, lo), value / (hi - lo)), 1);
        }
        const p = std.abs(
            std.sub(std.mul(std.fract(std.add(d.vec3f(uv.y * 2), d.vec3f(1, 2 / 3, 1 / 3))), 6), 3),
        );
        return d.vec4f(
            std.mul(
                std.clamp(std.sub(p, 1), d.vec3f(0), d.vec3f(1)),
                std.max(0, std.exp2(uv.x * 9) - 1),
            ),
            1,
        );
    },
});
export const PatternPlugin = MaterialPlugin(PatternMaterial);
