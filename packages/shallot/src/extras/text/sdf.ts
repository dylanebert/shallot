import tgpu, { type TgpuRenderPipeline } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { Compute } from "../../engine";

// The glyph SDF generator: two raster passes per glyph, both authored in TGSL over explicit bind group
// layouts. The distance pass rasterizes one instanced fullscreen triangle per outline segment into a
// shared `rgba8unorm` intermediate, additively accumulating an exponent-encoded distance falloff in
// alpha (blend op `max`, so the nearest segment wins) and the even-odd winding crossings in r/g. The
// finalize pass reads that intermediate, flips the falloff where the crossings disagree (inside), and
// writes the single-channel result into one `sdfSize` tile of the caller's atlas — placed by the pass's
// viewport + scissor rect, which is why the finalize draw goes through a pass this file owns.

/** the SDF falloff exponent the atlas bakes and {@link sdfToSignedDistance} inverts. @internal */
export const SDF_EXPONENT = 9;

interface LineSegment {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
}

function pointOnQuadraticBezier(
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    t: number,
): { x: number; y: number } {
    const t2 = 1 - t;
    return {
        x: t2 * t2 * x0 + 2 * t2 * t * x1 + t * t * x2,
        y: t2 * t2 * y0 + 2 * t2 * t * y1 + t * t * y2,
    };
}

function pointOnCubicBezier(
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    x3: number,
    y3: number,
    t: number,
): { x: number; y: number } {
    const t2 = 1 - t;
    return {
        x: t2 * t2 * t2 * x0 + 3 * t2 * t2 * t * x1 + 3 * t2 * t * t * x2 + t * t * t * x3,
        y: t2 * t2 * t2 * y0 + 3 * t2 * t2 * t * y1 + 3 * t2 * t * t * y2 + t * t * t * y3,
    };
}

export function segmentPath(pathString: string, curvePoints = 16): LineSegment[] {
    const segments: LineSegment[] = [];
    const segmentRE = /([MLQCZ])([^MLQCZ]*)/g;
    let match: RegExpExecArray | null;
    let firstX = 0,
        firstY = 0,
        prevX = 0,
        prevY = 0;

    while ((match = segmentRE.exec(pathString))) {
        const args = match[2]
            .trim()
            .split(/[,\s]+/)
            .filter((s) => s)
            .map((v) => parseFloat(v));

        switch (match[1]) {
            case "M":
                prevX = firstX = args[0];
                prevY = firstY = args[1];
                break;
            case "L":
                if (args[0] !== prevX || args[1] !== prevY) {
                    segments.push({ x1: prevX, y1: prevY, x2: args[0], y2: args[1] });
                }
                prevX = args[0];
                prevY = args[1];
                break;
            case "Q": {
                let curveX = prevX;
                let curveY = prevY;
                for (let i = 1; i < curvePoints; i++) {
                    const pt = pointOnQuadraticBezier(
                        prevX,
                        prevY,
                        args[0],
                        args[1],
                        args[2],
                        args[3],
                        i / (curvePoints - 1),
                    );
                    segments.push({ x1: curveX, y1: curveY, x2: pt.x, y2: pt.y });
                    curveX = pt.x;
                    curveY = pt.y;
                }
                prevX = args[2];
                prevY = args[3];
                break;
            }
            case "C": {
                let curveX = prevX;
                let curveY = prevY;
                for (let i = 1; i < curvePoints; i++) {
                    const pt = pointOnCubicBezier(
                        prevX,
                        prevY,
                        args[0],
                        args[1],
                        args[2],
                        args[3],
                        args[4],
                        args[5],
                        i / (curvePoints - 1),
                    );
                    segments.push({ x1: curveX, y1: curveY, x2: pt.x, y2: pt.y });
                    curveX = pt.x;
                    curveY = pt.y;
                }
                prevX = args[4];
                prevY = args[5];
                break;
            }
            case "Z":
                if (prevX !== firstX || prevY !== firstY) {
                    segments.push({ x1: prevX, y1: prevY, x2: firstX, y2: firstY });
                }
                prevX = firstX;
                prevY = firstY;
                break;
        }
    }

    return segments;
}

/** the per-glyph distance-pass uniform: the padded glyph bounds the triangle rasterizes across (xy = min,
 *  zw = max), the max distance the falloff normalizes by, and the SDF exponent. @internal */
export const SdfUniforms = d.struct({
    glyphBounds: d.vec4f,
    maxDistance: d.f32,
    exponent: d.f32,
    pad: d.vec2f,
});

/** the distance pass's I/O: the per-glyph uniform + the glyph's flattened outline segments (each
 *  `(x1, y1, x2, y2)`), one instance per segment. The stages are pinned rather than defaulted —
 *  `segments` reaches only the fs, and a vertex-visible storage binding is what a device reporting
 *  `maxStorageBuffersInVertexStage: 0` rejects (the `layout: "auto"` this replaced derived the same
 *  narrow visibility from use). @internal */
export const distanceLayout = tgpu.bindGroupLayout({
    uniforms: { uniform: SdfUniforms, visibility: ["vertex", "fragment"] },
    segments: {
        storage: (n: number) => d.arrayOf(d.vec4f, n),
        access: "readonly",
        visibility: ["fragment"],
    },
});

/** the finalize pass's I/O: the accumulated intermediate + the nearest-filter sampler. @internal */
export const finalizeLayout = tgpu.bindGroupLayout({
    intermediate: { texture: d.texture2d(d.f32), visibility: ["fragment"] },
    samp: { sampler: "non-filtering", visibility: ["fragment"] },
});

// a fullscreen triangle covering the viewport, uv 0..1 across the covered quad — the same three-vertex
// form the mipmap blit uses (`standard/render/image.ts`)
const fullscreenUv = tgpu.fn(
    [d.u32],
    d.vec2f,
)((vid) => {
    "use gpu";
    return d.vec2f(d.f32((vid << 1) & 2), d.f32(vid & 2));
});

const distanceVs = tgpu.vertexFn({
    in: { vid: d.builtin.vertexIndex, segmentIdx: d.builtin.instanceIndex },
    out: {
        pos: d.builtin.position,
        glyphXY: d.vec2f,
        segmentIdx: d.interpolate("flat", d.u32),
    },
})((input) => {
    "use gpu";
    const uv = fullscreenUv(input.vid);
    const b = distanceLayout.$.uniforms.glyphBounds;
    return {
        pos: d.vec4f(uv.x * 2 - 1, uv.y * 2 - 1, 0, 1),
        glyphXY: std.mix(b.xy, b.zw, uv),
        segmentIdx: input.segmentIdx,
    };
});

// one segment's contribution at this glyph-space point: an exponent-encoded distance falloff in alpha
// (blended `max`, so the nearest segment wins) plus the even-odd crossing counters in r/g (blended
// `add`, one 1/255 step per crossing — the finalize pass compares the two to classify inside/outside)
const distanceFs = tgpu.fragmentFn({
    in: { glyphXY: d.vec2f, segmentIdx: d.interpolate("flat", d.u32) },
    out: d.vec4f,
})((input) => {
    "use gpu";
    const seg = distanceLayout.$.segments[input.segmentIdx];
    const p = input.glyphXY;

    const lineDir = std.sub(seg.zw, seg.xy);
    const lenSq = std.dot(lineDir, lineDir);
    // a degenerate segment divides 0/0; both select arms evaluate, and the NaN is discarded by the cond
    const t = std.select(
        0,
        std.clamp(std.dot(std.sub(p, seg.xy), lineDir) / lenSq, 0, 1),
        lenSq > 0,
    );
    const closest = std.add(seg.xy, std.mul(t, lineDir));
    const dist = std.distance(p, closest);

    const val =
        std.pow(
            1 - std.clamp(dist / distanceLayout.$.uniforms.maxDistance, 0, 1),
            distanceLayout.$.uniforms.exponent,
        ) * 0.5;

    const crosses = seg.y > p.y !== seg.w > p.y;
    const crossX = ((seg.z - seg.x) * (p.y - seg.y)) / (seg.w - seg.y) + seg.x;
    const crossingUp = crosses && p.x < crossX && seg.y < seg.w;
    const crossingDown = crosses && p.x < crossX && seg.y > seg.w;

    return d.vec4f(
        std.select(0, 1 / 255, crossingUp),
        std.select(0, 1 / 255, crossingDown),
        0,
        val,
    );
});

const finalizeVs = tgpu.vertexFn({
    in: { vid: d.builtin.vertexIndex },
    out: { pos: d.builtin.position, uv: d.vec2f },
})((input) => {
    "use gpu";
    const uv = fullscreenUv(input.vid);
    return { pos: d.vec4f(uv.x * 2 - 1, uv.y * 2 - 1, 0, 1), uv: uv };
});

// inside ⇔ the two crossing counters disagree (odd winding), which flips the falloff
const finalizeFs = tgpu.fragmentFn({
    in: { uv: d.vec2f },
    out: d.vec4f,
})((input) => {
    "use gpu";
    const color = std.textureSample(finalizeLayout.$.intermediate, finalizeLayout.$.samp, input.uv);
    const inside = color.x !== color.y;
    const val = std.select(color.w, 1 - color.w, inside);
    return d.vec4f(val, val, val, val);
});

/** the emitted SDF-pass WGSL — the device-free structural seam its test resolves.
 *  @internal */
export function sdfWgsl(): { distance: string; finalize: string } {
    return {
        distance: tgpu.resolve([distanceVs, distanceFs], { names: "strict" }),
        finalize: tgpu.resolve([finalizeVs, finalizeFs], { names: "strict" }),
    };
}

// pipelines come from `Compute.root`, which is device-scoped, so a stale entry from a torn-down device
// must not be reused — keyed by the device identity it was built against, like the mipmap blit's cache.
// Shared across generators (one per font), which the per-instance pair they replaced was not.
let _pipelines: {
    device: GPUDevice;
    distance: TgpuRenderPipeline;
    finalize: TgpuRenderPipeline;
} | null = null;

function pipelines(device: GPUDevice) {
    if (_pipelines && _pipelines.device === device) return _pipelines;
    // the generator's own resources (sampler, intermediate target, encoder) still come from the passed
    // device; only the pipelines are root-bound, and `Compute.root` is scoped to `Compute.device`. So a
    // foreign device would silently mix two devices in one pass. One root per device lands at 4a with
    // `image.ts`'s identical narrowing (the typed extension surface); until then the contract is explicit
    // rather than silent
    if (device !== Compute.device)
        throw new Error(
            "[text] the SDF generator builds its pipelines on Compute.device — pass that device, " +
                "or call requestGPU with yours first",
        );
    const root = Compute.root;
    _pipelines = {
        device,
        distance: root
            .createRenderPipeline({
                vertex: distanceVs,
                fragment: distanceFs,
                targets: {
                    format: "rgba8unorm",
                    blend: {
                        // alpha accumulates as `max` — the nearest segment's falloff wins; r/g add up
                        // one 1/255 step per winding crossing
                        color: { srcFactor: "one", dstFactor: "one", operation: "add" },
                        alpha: { srcFactor: "one", dstFactor: "one", operation: "max" },
                    },
                },
                primitive: { topology: "triangle-list" },
            })
            .$name("text-sdf-distance"),
        finalize: root
            .createRenderPipeline({
                vertex: finalizeVs,
                fragment: finalizeFs,
                targets: { format: "r8unorm" },
                primitive: { topology: "triangle-list" },
            })
            .$name("text-sdf-finalize"),
    };
    return _pipelines;
}

export interface SDFGeneratorConfig {
    device: GPUDevice;
    sdfSize?: number;
    exponent?: number;
    curveSubdivisions?: number;
}

export class SDFGenerator {
    private _device: GPUDevice;
    private _sdfSize: number;
    private _exponent: number;
    private _curveSubdivisions: number;

    private _pipelines: ReturnType<typeof pipelines> | null = null;

    private _intermediateTexture: GPUTexture | null = null;
    private _sampler: GPUSampler;

    private _maxSegments = 4096;
    private _pending: {
        path: string;
        bounds: [number, number, number, number];
        outputTexture: GPUTexture;
        outputX: number;
        outputY: number;
    }[] = [];

    constructor(config: SDFGeneratorConfig) {
        this._device = config.device;
        this._sdfSize = config.sdfSize ?? 64;
        this._exponent = config.exponent ?? SDF_EXPONENT;
        this._curveSubdivisions = config.curveSubdivisions ?? 16;

        this._sampler = this._device.createSampler({
            magFilter: "nearest",
            minFilter: "nearest",
        });
    }

    private ensureIntermediateTexture(): void {
        if (this._intermediateTexture) return;

        this._intermediateTexture = this._device.createTexture({
            size: { width: this._sdfSize, height: this._sdfSize },
            format: "rgba8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
    }

    begin(): void {
        // built here, drawn from `flush` microseconds later, so there is no force-compile forcer: a
        // `precompile` thunk drains after warm, long after these draws already went out (the load-path
        // blit's refuted precompile, spec Approach 2a)
        this._pipelines = pipelines(this._device);
        this.ensureIntermediateTexture();
        this._pending = [];
    }

    add(
        path: string,
        bounds: [number, number, number, number],
        outputTexture: GPUTexture,
        outputX: number,
        outputY: number,
    ): void {
        this._pending.push({ path, bounds, outputTexture, outputX, outputY });
    }

    flush(): void {
        if (this._pending.length === 0) return;
        const pipes = this._pipelines;
        const intermediate = this._intermediateTexture;
        if (!pipes || !intermediate) throw new Error("[text] SDFGenerator.flush before begin");

        const root = Compute.root;
        const encoder = this._device.createCommandEncoder({ label: "text-sdf" });
        // one uniform + one segment buffer per glyph, all read within the single submit below — a shared
        // buffer overwritten per glyph would hand every pass the last glyph's data, since the queue
        // writes all land before the submit. Destroyed once that submit is in flight.
        const tempBuffers: { destroy(): void }[] = [];

        for (const entry of this._pending) {
            const segments = segmentPath(entry.path, this._curveSubdivisions);
            if (segments.length === 0) continue;
            if (segments.length > this._maxSegments) {
                console.warn(
                    `Too many segments (${segments.length}), truncating to ${this._maxSegments}`,
                );
                segments.length = this._maxSegments;
            }

            const [xMin, yMin, xMax, yMax] = entry.bounds;
            const maxDist = Math.max(xMax - xMin, yMax - yMin) / 2;

            // the uniform writes through the schema, so no lane index is hand-counted
            const uniformBuffer = root
                .createBuffer(SdfUniforms, {
                    glyphBounds: d.vec4f(xMin, yMin, xMax, yMax),
                    maxDistance: maxDist,
                    exponent: this._exponent,
                    pad: d.vec2f(),
                })
                .$usage("uniform")
                .$name("text-sdf-uniforms");
            tempBuffers.push(uniformBuffer);

            // the ArrayBuffer write path, never the schema-array form (gpu.md: the serializer is ~480×
            // `Float32Array.set`) — CPU truth stays a typed array
            const segmentData = new Float32Array(segments.length * 4);
            for (let i = 0; i < segments.length; i++) {
                const seg = segments[i];
                segmentData[i * 4] = seg.x1;
                segmentData[i * 4 + 1] = seg.y1;
                segmentData[i * 4 + 2] = seg.x2;
                segmentData[i * 4 + 3] = seg.y2;
            }
            const segmentBuffer = root
                .createBuffer(d.arrayOf(d.vec4f, segments.length))
                .$usage("storage")
                .$name("text-sdf-segments");
            segmentBuffer.write(segmentData.buffer);
            tempBuffers.push(segmentBuffer);

            const distanceGroup = root.createBindGroup(distanceLayout, {
                uniforms: uniformBuffer,
                segments: segmentBuffer,
            });

            const distancePass = encoder.beginRenderPass({
                label: "text-sdf-distance",
                colorAttachments: [
                    {
                        view: intermediate.createView(),
                        clearValue: { r: 0, g: 0, b: 0, a: 0 },
                        loadOp: "clear",
                        storeOp: "store",
                    },
                ],
                timestampWrites: Compute.span?.("text:sdf-distance"),
            });

            pipes.distance.with(distanceGroup).with(distancePass).draw(3, segments.length);
            distancePass.end();

            const finalizeGroup = root.createBindGroup(finalizeLayout, {
                intermediate: intermediate.createView(),
                samp: this._sampler,
            });

            const outputView = entry.outputTexture.createView({
                baseMipLevel: 0,
                mipLevelCount: 1,
                baseArrayLayer: 0,
                arrayLayerCount: 1,
            });

            const finalizePass = encoder.beginRenderPass({
                label: "text-sdf-finalize",
                colorAttachments: [
                    {
                        view: outputView,
                        loadOp: "load",
                        storeOp: "store",
                    },
                ],
                timestampWrites: Compute.span?.("text:sdf-finalize"),
            });

            // the viewport + scissor are what place this glyph in its atlas tile, and a typed pipeline
            // exposes neither — `.with(pass)` records into the pass this file already owns, so they stay
            // raw calls on it, set before the draw
            finalizePass.setViewport(
                entry.outputX,
                entry.outputY,
                this._sdfSize,
                this._sdfSize,
                0,
                1,
            );
            finalizePass.setScissorRect(entry.outputX, entry.outputY, this._sdfSize, this._sdfSize);
            pipes.finalize.with(finalizeGroup).with(finalizePass).draw(3);
            finalizePass.end();
        }

        this._device.queue.submit([encoder.finish()]);

        for (const buffer of tempBuffers) buffer.destroy();
        this._pending = [];
    }

    destroy(): void {
        this._intermediateTexture?.destroy();
    }
}
