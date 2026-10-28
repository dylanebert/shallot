import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as s from "typegpu/std";
import { ViewUniforms } from "../../core/rendering";

// Visibility bitmask AO (Therrien et al., 2023), Bevy b3bdd75be
// ssao/{preprocess_depth,ssao,spatial_denoise}.wesl.
export const SsaoSettings = d.struct({
    radius: d.f32,
    thickness: d.f32,
    slices: d.u32,
    samples: d.u32,
});
const common = {
    view: { uniform: ViewUniforms },
    settings: { uniform: SsaoSettings },
} as const;
export const copyLayout = tgpu.bindGroupLayout({
    depth: { texture: d.textureDepth2d() },
    normals: { texture: d.texture2d() },
    outputDepth: { storageTexture: d.textureStorage2d("rgba16float", "write-only") },
    outputNormals: { storageTexture: d.textureStorage2d("rgba16float", "write-only") },
});
export const copyMsaaLayout = tgpu.bindGroupLayout({
    depth: { texture: d.textureDepthMultisampled2d() },
    normals: { texture: d.textureMultisampled2d(), sampleType: "unfilterable-float" },
    outputDepth: { storageTexture: d.textureStorage2d("rgba16float", "write-only") },
    outputNormals: { storageTexture: d.textureStorage2d("rgba16float", "write-only") },
});
const copyDepth = tgpu.fn(
    [d.vec2i],
    d.f32,
)((px) => {
    "use gpu";
    return s.textureLoad(copyLayout.$.depth, px, 0);
});
const copyDepthMsaa = tgpu.fn(
    [d.vec2i],
    d.f32,
)((px) => {
    "use gpu";
    return s.textureLoad(copyMsaaLayout.$.depth, px, 0);
});
const copyNormal = tgpu.fn(
    [d.vec2i],
    d.vec4f,
)((px) => {
    "use gpu";
    return s.textureLoad(copyLayout.$.normals, px, 0);
});
const copyNormalMsaa = tgpu.fn(
    [d.vec2i],
    d.vec4f,
)((px) => {
    "use gpu";
    return s.textureLoad(copyMsaaLayout.$.normals, px, 0);
});
export function copyKernel(multisampled: boolean) {
    const layout = multisampled ? copyMsaaLayout : copyLayout;
    const loadDepth = multisampled ? copyDepthMsaa : copyDepth;
    const loadNormal = multisampled ? copyNormalMsaa : copyNormal;
    return tgpu.computeFn({ workgroupSize: [8, 8], in: { id: d.builtin.globalInvocationId } })(
        (input) => {
            "use gpu";
            const px = d.vec2i(input.id.xy);
            if (s.any(s.ge(input.id.xy, s.textureDimensions(layout.$.outputDepth)))) return;
            s.textureStore(layout.$.outputDepth, px, d.vec4f(loadDepth(px), 0, 0, 0));
            s.textureStore(layout.$.outputNormals, px, loadNormal(px));
        },
    );
}
export const mipLayout = tgpu.bindGroupLayout({
    depth: { texture: d.texture2d() },
    output: { storageTexture: d.textureStorage2d("rgba16float", "write-only") },
});
const weightedAverage = tgpu.fn(
    [d.vec4f],
    d.f32,
)((depths) => {
    "use gpu";
    const radius = 0.75 * 0.5 * 1.457;
    const falloffRange = 0.615 * radius;
    const falloffFrom = radius * (1 - 0.615);
    const least = s.min(s.min(depths.x, depths.y), s.min(depths.z, depths.w));
    const weights = s.saturate(
        s.add(s.mul(s.sub(depths, least), -1 / falloffRange), falloffFrom / falloffRange + 1),
    );
    return s.dot(weights, depths) / s.dot(weights, d.vec4f(1));
});
export const mipKernel = tgpu.computeFn({
    workgroupSize: [8, 8],
    in: { id: d.builtin.globalInvocationId },
})((input) => {
    "use gpu";
    const px = d.vec2i(input.id.xy);
    if (s.any(s.ge(input.id.xy, s.textureDimensions(mipLayout.$.output)))) return;
    const base = s.mul(px, 2);
    const last = s.sub(d.vec2i(s.textureDimensions(mipLayout.$.depth)), 1);
    const depths = d.vec4f(
        s.textureLoad(mipLayout.$.depth, s.min(base, last), 0).x,
        s.textureLoad(mipLayout.$.depth, s.min(s.add(base, d.vec2i(1, 0)), last), 0).x,
        s.textureLoad(mipLayout.$.depth, s.min(s.add(base, d.vec2i(0, 1)), last), 0).x,
        s.textureLoad(mipLayout.$.depth, s.min(s.add(base, d.vec2i(1)), last), 0).x,
    );
    s.textureStore(mipLayout.$.output, px, d.vec4f(weightedAverage(depths), 0, 0, 0));
});
export const aoLayout = tgpu.bindGroupLayout({
    ...common,
    depth: { texture: d.texture2d() },
    depthSampler: { sampler: "filtering" },
    normals: { texture: d.texture2d() },
    noise: { texture: d.texture2d(d.u32) },
    output: { storageTexture: d.textureStorage2d("r32float", "write-only") },
    edges: { storageTexture: d.textureStorage2d("r32uint", "write-only") },
});
const depthAt = tgpu.fn(
    [d.vec2i, d.i32],
    d.f32,
)((px, mip) => {
    "use gpu";
    const last = s.sub(d.vec2i(s.textureDimensions(aoLayout.$.depth, mip)), 1);
    return s.textureLoad(aoLayout.$.depth, s.clamp(px, d.vec2i(0), last), mip).x;
});
const position = tgpu.fn(
    [d.f32, d.vec2f],
    d.vec3f,
)((depth, uv) => {
    "use gpu";
    const clip = d.vec4f(uv.x * 2 - 1, 1 - 2 * uv.y, depth, 1);
    const world = s.mul(aoLayout.$.view.worldFromClip, clip);
    return s.mul(aoLayout.$.view.viewFromWorld, d.vec4f(s.div(world.xyz, world.w), 1)).xyz;
});
const samplePosition = tgpu.fn(
    [d.vec2f, d.f32],
    d.vec3f,
)((uv, mip) => {
    "use gpu";
    return position(s.textureSampleLevel(aoLayout.$.depth, aoLayout.$.depthSampler, uv, mip).x, uv);
});
const fastAcos = tgpu.fn(
    [d.f32],
    d.f32,
)((x) => {
    "use gpu";
    const a = s.abs(x);
    const result = (-0.156583 * a + Math.PI / 2) * s.sqrt(1 - a);
    return s.select(Math.PI - result, result, x >= 0);
});
const processSample = tgpu.fn(
    [d.vec3f, d.vec3f, d.f32, d.f32, d.f32, d.u32],
    d.u32,
)((delta, view, direction, n, samples, mask) => {
    "use gpu";
    const back = s.sub(delta, s.mul(view, aoLayout.$.settings.thickness));
    let horizons = d.vec2f(
        fastAcos(s.dot(s.normalize(delta), view)),
        fastAcos(s.dot(s.normalize(back), view)),
    );
    horizons = s.saturate(s.fma(d.vec2f(direction), s.mul(horizons, -1 / Math.PI), d.vec2f(n)));
    horizons = s.select(horizons.xy, horizons.yx, direction >= 0);
    const start = d.u32(horizons.x * samples);
    const count = d.u32(s.ceil((horizons.y - horizons.x) * samples));
    return s.insertBits(mask, d.u32(0xffffffff), start, count);
});
export function aoKernel() {
    return tgpu.computeFn({ workgroupSize: [8, 8], in: { id: d.builtin.globalInvocationId } })(
        (input) => {
            "use gpu";
            const px = d.vec2i(input.id.xy);
            if (s.any(s.ge(input.id.xy, s.textureDimensions(aoLayout.$.output)))) return;
            const slices = d.f32(aoLayout.$.settings.slices);
            const samples = d.f32(aoLayout.$.settings.samples);
            const size = aoLayout.$.view.resolution;
            const uv = s.div(s.add(d.vec2f(px), 0.5), size);
            const center = depthAt(px, 0);
            let edges = s.sub(
                d.vec4f(
                    depthAt(s.add(px, d.vec2i(-1, 0)), 0),
                    depthAt(s.add(px, d.vec2i(1, 0)), 0),
                    depthAt(s.add(px, d.vec2i(0, -1)), 0),
                    depthAt(s.add(px, d.vec2i(0, 1)), 0),
                ),
                center,
            );
            const lr = (edges.y - edges.x) * 0.5;
            const tb = (edges.w - edges.z) * 0.5;
            edges = s.min(s.abs(edges), s.abs(s.add(edges, d.vec4f(lr, -lr, tb, -tb))));
            edges = s.saturate(s.sub(1.25, s.div(edges, center * 0.011)));
            s.textureStore(aoLayout.$.edges, px, d.vec4u(s.pack4x8unorm(edges), 0, 0, 0));
            const p = position(center + 0.00001, uv);
            const worldNormal = s.sub(s.mul(s.textureLoad(aoLayout.$.normals, px, 0).xyz, 2), 1);
            const normal = s.mul(aoLayout.$.view.viewFromWorld, d.vec4f(worldNormal, 0)).xyz;
            const view = s.normalize(s.neg(p));
            const index = s.textureLoad(aoLayout.$.noise, d.vec2i(px.x % 64, px.y % 64), 0).x;
            const noise = s.fract(
                s.add(0.5, s.mul(d.f32(index), d.vec2f(0.7548776662466928, 0.5698402909980532))),
            );
            const scale =
                (-0.5 *
                    s.max(aoLayout.$.settings.radius, 0.0001) *
                    s.mul(aoLayout.$.view.clipFromView, d.vec4f(1, 0, 0, 0)).x) /
                p.z;
            let occluded = d.u32(0);
            for (let slice = d.f32(0); slice < slices; slice += 1) {
                const phi = (Math.PI / slices) * (slice + noise.x);
                const omega = d.vec2f(s.cos(phi), s.sin(phi));
                const direction = d.vec3f(omega, 0);
                const ortho = s.sub(direction, s.mul(s.dot(direction, view), view));
                const axis = s.cross(direction, view);
                const projected = s.sub(normal, s.mul(axis, s.dot(normal, axis)));
                const sign = s.sign(s.dot(ortho, projected));
                const cos = s.saturate(s.dot(projected, view) / s.length(projected));
                const n = (Math.PI / 2 - sign * fastAcos(cos)) / Math.PI;
                let mask = d.u32(0);
                const mul = s.mul(d.vec2f(omega.x, -omega.y), scale);
                for (let sample = d.f32(0); sample < samples; sample += 1) {
                    const noiseSample = s.fract(
                        noise.y + (slice + sample * samples) * 0.6180339887498948,
                    );
                    let t = (sample + noiseSample) / samples;
                    t *= t;
                    const offset = s.mul(t, mul);
                    const mip = s.clamp(
                        s.log2(s.length(s.mul(offset, size))) - 3.3,
                        0,
                        d.f32(s.textureNumLevels(aoLayout.$.depth) - 1),
                    );
                    mask = processSample(
                        s.sub(samplePosition(s.add(uv, offset), mip), p),
                        view,
                        -1,
                        n,
                        samples * 2,
                        mask,
                    );
                    mask = processSample(
                        s.sub(samplePosition(s.sub(uv, offset), mip), p),
                        view,
                        1,
                        n,
                        samples * 2,
                        mask,
                    );
                }
                occluded += s.countOneBits(mask);
            }
            const visibility = s.clamp(1 - d.f32(occluded) / (slices * 2 * samples), 0.03, 1);
            s.textureStore(aoLayout.$.output, px, d.vec4f(visibility, 0, 0, 0));
        },
    );
}
export const denoiseLayout = tgpu.bindGroupLayout({
    noisy: { texture: d.texture2d(), sampleType: "unfilterable-float" },
    edges: { texture: d.texture2d(d.u32) },
    output: { storageTexture: d.textureStorage2d("r32float", "write-only") },
});
const edgeAt = tgpu.fn(
    [d.vec2i],
    d.vec4f,
)((px) => {
    "use gpu";
    const last = s.sub(d.vec2i(s.textureDimensions(denoiseLayout.$.edges)), 1);
    return s.unpack4x8unorm(
        s.textureLoad(denoiseLayout.$.edges, s.clamp(px, d.vec2i(0), last), 0).x,
    );
});
const visibilityAt = tgpu.fn(
    [d.vec2i],
    d.f32,
)((px) => {
    "use gpu";
    const last = s.sub(d.vec2i(s.textureDimensions(denoiseLayout.$.noisy)), 1);
    return s.textureLoad(denoiseLayout.$.noisy, s.clamp(px, d.vec2i(0), last), 0).x;
});
export const denoiseKernel = tgpu.computeFn({
    workgroupSize: [8, 8],
    in: { id: d.builtin.globalInvocationId },
})((input) => {
    "use gpu";
    const px = d.vec2i(input.id.xy);
    if (s.any(s.ge(input.id.xy, s.textureDimensions(denoiseLayout.$.output)))) return;
    const left = edgeAt(s.add(px, d.vec2i(-1, 0)));
    const right = edgeAt(s.add(px, d.vec2i(1, 0)));
    const top = edgeAt(s.add(px, d.vec2i(0, -1)));
    const bottom = edgeAt(s.add(px, d.vec2i(0, 1)));
    const edges = s.mul(edgeAt(px), d.vec4f(left.y, right.x, top.w, bottom.z));
    const tl = 0.425 * (edges.z * top.x + edges.x * left.z);
    const tr = 0.425 * (edges.z * top.y + edges.y * right.z);
    const bl = 0.425 * (edges.w * bottom.x + edges.x * left.w);
    const br = 0.425 * (edges.w * bottom.y + edges.y * right.w);
    let sum = visibilityAt(px) * 1.2;
    sum += visibilityAt(s.add(px, d.vec2i(-1, 0))) * edges.x;
    sum += visibilityAt(s.add(px, d.vec2i(1, 0))) * edges.y;
    sum += visibilityAt(s.add(px, d.vec2i(0, -1))) * edges.z;
    sum += visibilityAt(s.add(px, d.vec2i(0, 1))) * edges.w;
    sum += visibilityAt(s.add(px, d.vec2i(-1, -1))) * tl;
    sum += visibilityAt(s.add(px, d.vec2i(1, -1))) * tr;
    sum += visibilityAt(s.add(px, d.vec2i(-1, 1))) * bl;
    sum += visibilityAt(s.add(px, d.vec2i(1, 1))) * br;
    const weight = 1.2 + edges.x + edges.y + edges.z + edges.w + tl + tr + bl + br;
    s.textureStore(denoiseLayout.$.output, px, d.vec4f(sum / weight, 0, 0, 0));
});
