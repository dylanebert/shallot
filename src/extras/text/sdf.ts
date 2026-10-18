import tgpu, { type TgpuRenderPipeline } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import type { World } from "../../engine";

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

/** Flatten an SVG outline path into line segments, `curvePoints` per curve. @internal */
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

/** one glyph's SDF uniform: the padded glyph bounds its cell covers (xy = min, zw = max), the max
 *  distance the falloff normalizes by, the SDF exponent, and the cutoff distance beyond which the
 *  falloff rounds to zero in the 8-bit atlas. */
const SdfUniforms = d.struct({
    glyphBounds: d.vec4f,
    maxDistance: d.f32,
    exponent: d.f32,
    cutoff: d.f32,
    pad: d.f32,
});

/** a run of at most {@link CHUNK} consecutive segments along one contour: their bounds, the run's
 *  start point and end height, and its segment range (first << 16 | count). */
const SdfChunk = d.struct({
    box: d.vec4f,
    start: d.vec2f,
    endY: d.f32,
    range: d.u32,
});
const CHUNK = 16;

/** one glyph's SDF I/O: its uniform, its flattened outline segments (each `(x1, y1, x2, y2)`) and
 *  their chunks. The stages are pinned rather than defaulted — the storage reaches only the fs, and
 *  a vertex-visible storage binding is what a device reporting `maxStorageBuffersInVertexStage: 0`
 *  rejects. */
const sdfLayout = tgpu.bindGroupLayout({
    uniforms: { uniform: SdfUniforms, visibility: ["vertex", "fragment"] },
    segments: {
        storage: (n: number) => d.arrayOf(d.vec4f, n),
        access: "readonly",
        visibility: ["fragment"],
    },
    chunks: {
        storage: (n: number) => d.arrayOf(SdfChunk, n),
        access: "readonly",
        visibility: ["fragment"],
    },
});

const sdfVs = tgpu.vertexFn({
    in: { vid: d.builtin.vertexIndex },
    out: { pos: d.builtin.position, glyphXY: d.vec2f },
})((input) => {
    "use gpu";
    const uv = d.vec2f(d.f32((input.vid << 1) & 2), d.f32(input.vid & 2));
    const b = sdfLayout.$.uniforms.glyphBounds;
    return {
        pos: d.vec4f(uv.x * 2 - 1, uv.y * 2 - 1, 0, 1),
        glyphXY: std.mix(b.xy, b.zw, d.vec2f(uv.x, 1 - uv.y)),
    };
});

// Each texel takes the distance to its nearest segment, capped at the cutoff, and classifies itself
// inside by the nonzero winding of a ray toward +x; inside flips the exponent-encoded falloff.
// Chunks bound the work: a chunk whose box is farther than the nearest distance so far is skipped,
// and a chunk wholly right of the texel adds its crossings from its endpoints alone, since each
// segment's signed crossing is [end above] - [start above] and a contiguous run telescopes.
const sdfFs = tgpu.fragmentFn({
    in: { glyphXY: d.vec2f },
    out: d.vec4f,
})((input) => {
    "use gpu";
    const p = input.glyphXY;
    const chunkCount = std.arrayLength(sdfLayout.$.chunks);
    // Start from the nearest chunk start: every start lies on the outline.
    let nearest = sdfLayout.$.uniforms.cutoff;
    for (let c = d.u32(0); c < chunkCount; c++) {
        nearest = std.min(nearest, std.distance(p, sdfLayout.$.chunks[c].start));
    }
    let winding = d.i32(0);
    for (let c = d.u32(0); c < chunkCount; c++) {
        const chunk = SdfChunk(sdfLayout.$.chunks[c]);
        const box = chunk.box;
        const level = p.y >= box.y && p.y <= box.w;
        if (level && p.x < box.x) {
            winding += d.i32(chunk.endY > p.y) - d.i32(chunk.start.y > p.y);
        }
        const straddles = level && p.x >= box.x && p.x <= box.z;
        const gap = std.max(std.max(std.sub(box.xy, p), std.sub(p, box.zw)), d.vec2f(0));
        if (straddles || std.length(gap) < nearest) {
            const first = chunk.range >>> 16;
            const end = first + (chunk.range & 0xffff);
            for (let i = first; i < end; i++) {
                const seg = d.vec4f(sdfLayout.$.segments[i]);
                const lineDir = std.sub(seg.zw, seg.xy);
                const lenSq = std.dot(lineDir, lineDir);
                // a degenerate segment divides 0/0; both select arms evaluate, and the cond discards the NaN
                const t = std.select(
                    0,
                    std.clamp(std.dot(std.sub(p, seg.xy), lineDir) / lenSq, 0, 1),
                    lenSq > 0,
                );
                nearest = std.min(nearest, std.distance(p, std.add(seg.xy, std.mul(t, lineDir))));
                if (straddles && seg.y > p.y !== seg.w > p.y) {
                    const crossX = ((seg.z - seg.x) * (p.y - seg.y)) / (seg.w - seg.y) + seg.x;
                    if (p.x < crossX) winding += std.select(d.i32(-1), d.i32(1), seg.y < seg.w);
                }
            }
        }
    }
    const falloff =
        std.pow(
            1 - std.clamp(nearest / sdfLayout.$.uniforms.maxDistance, 0, 1),
            sdfLayout.$.uniforms.exponent,
        ) * 0.5;
    const val = std.select(falloff, 1 - falloff, winding !== 0);
    return d.vec4f(val, val, val, val);
});

// Each World caches one pipeline from its own TypeGPU root. Its font generators share it; another
// World gets its own pipeline even when both Worlds use the same GPUDevice.
const sdfPipelineKey = { create: () => ({ value: null as TgpuRenderPipeline | null }) };

function pipeline(world: World): TgpuRenderPipeline {
    const cache = world.resource(sdfPipelineKey);
    cache.value ??= world.gpu.root
        .createRenderPipeline({
            vertex: sdfVs,
            fragment: sdfFs,
            targets: { format: "r8unorm" },
            primitive: { topology: "triangle-list" },
        })
        .$name("text-sdf");
    return cache.value;
}

/** Drop this World's cached SDF pipeline. @internal */
export function resetPipelines(world: World): void {
    world.resource(sdfPipelineKey).value = null;
}

const CHUNK_BYTES = 32;

/** Pack segments into runs of at most {@link CHUNK}, breaking wherever one segment does not start
 *  at the previous one's end, so each run's crossings telescope. */
function chunkSegments(segments: readonly LineSegment[]): ArrayBuffer {
    const runs: [number, number][] = [];
    for (let i = 0; i < segments.length; i++) {
        const run = runs.at(-1);
        const prev = segments[i - 1];
        if (run && run[1] < CHUNK && prev.x2 === segments[i].x1 && prev.y2 === segments[i].y1)
            run[1]++;
        else runs.push([i, 1]);
    }
    const bytes = new ArrayBuffer(runs.length * CHUNK_BYTES);
    const f32 = new Float32Array(bytes);
    const u32 = new Uint32Array(bytes);
    for (let c = 0; c < runs.length; c++) {
        const [first, count] = runs[c];
        let minX = Infinity,
            minY = Infinity,
            maxX = -Infinity,
            maxY = -Infinity;
        for (let i = first; i < first + count; i++) {
            const seg = segments[i];
            minX = Math.min(minX, seg.x1, seg.x2);
            minY = Math.min(minY, seg.y1, seg.y2);
            maxX = Math.max(maxX, seg.x1, seg.x2);
            maxY = Math.max(maxY, seg.y1, seg.y2);
        }
        const at = c * 8;
        f32.set([minX, minY, maxX, maxY, segments[first].x1, segments[first].y1], at);
        f32[at + 6] = segments[first + count - 1].y2;
        u32[at + 7] = (first << 16) | count;
    }
    return bytes;
}

interface SDFGeneratorConfig {
    device: GPUDevice;
    sdfSize: number;
    exponent: number;
    curveSubdivisions: number;
}

export class SDFGenerator {
    private _device: GPUDevice;
    private _sdfSize: number;
    private _exponent: number;
    private _curveSubdivisions: number;

    private _pipeline: TgpuRenderPipeline | null = null;

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
        this._sdfSize = config.sdfSize;
        this._exponent = config.exponent;
        this._curveSubdivisions = config.curveSubdivisions;
    }

    begin(world: World): void {
        // flush draws during warm, before deferred precompile thunks drain.
        this._pipeline = pipeline(world);
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

    flush(world: World): void {
        if (this._pending.length === 0) return;
        const pipe = this._pipeline;
        if (!pipe) throw new Error("[text] SDFGenerator.flush before begin");

        const root = world.gpu.root;
        const encoder = this._device.createCommandEncoder({ label: "text-sdf" });
        // one uniform + one segment buffer per glyph, all read within the single submit below — a shared
        // buffer overwritten per glyph would hand every draw the last glyph's data, since the queue
        // writes all land before the submit. Destroyed once that submit is in flight.
        const tempBuffers: { destroy(): void }[] = [];
        let pass: GPURenderPassEncoder | null = null;
        let passTexture: GPUTexture | null = null;

        for (const entry of this._pending) {
            const segments = segmentPath(entry.path, this._curveSubdivisions);
            if (segments.length === 0) continue;
            if (segments.length > this._maxSegments) {
                console.warn(
                    `[text] too many segments (${segments.length}), truncating to ${this._maxSegments}`,
                );
                segments.length = this._maxSegments;
            }

            const [xMin, yMin, xMax, yMax] = entry.bounds;
            const maxDist = Math.max(xMax - xMin, yMax - yMin) / 2;

            const uniformBuffer = root
                .createBuffer(SdfUniforms, {
                    glyphBounds: d.vec4f(xMin, yMin, xMax, yMax),
                    maxDistance: maxDist,
                    exponent: this._exponent,
                    cutoff: maxDist * (1 - (1 / 510) ** (1 / this._exponent)),
                    pad: 0,
                })
                .$usage("uniform")
                .$name("text-sdf-uniforms");
            tempBuffers.push(uniformBuffer);

            // Write the ArrayBuffers to bypass schema-array serialization.
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
            const chunkData = chunkSegments(segments);
            const chunkBuffer = root
                .createBuffer(d.arrayOf(SdfChunk, chunkData.byteLength / CHUNK_BYTES))
                .$usage("storage")
                .$name("text-sdf-chunks");
            chunkBuffer.write(chunkData);
            tempBuffers.push(chunkBuffer);

            if (passTexture !== entry.outputTexture) {
                pass?.end();
                passTexture = entry.outputTexture;
                pass = encoder.beginRenderPass({
                    label: "text-sdf",
                    colorAttachments: [
                        {
                            view: entry.outputTexture.createView({
                                baseMipLevel: 0,
                                mipLevelCount: 1,
                                baseArrayLayer: 0,
                                arrayLayerCount: 1,
                            }),
                            loadOp: "load",
                            storeOp: "store",
                        },
                    ],
                    timestampWrites: world.gpu.span?.("text:sdf"),
                });
            }
            // Atlas placement needs viewport and scissor, which the pipeline API does not expose.
            pass!.setViewport(entry.outputX, entry.outputY, this._sdfSize, this._sdfSize, 0, 1);
            pass!.setScissorRect(entry.outputX, entry.outputY, this._sdfSize, this._sdfSize);
            pipe.with(
                root.createBindGroup(sdfLayout, {
                    uniforms: uniformBuffer,
                    segments: segmentBuffer,
                    chunks: chunkBuffer,
                }),
            )
                .with(pass!)
                .draw(3);
        }
        pass?.end();

        this._device.queue.submit([encoder.finish()]);

        for (const buffer of tempBuffers) buffer.destroy();
        this._pending = [];
    }
}
