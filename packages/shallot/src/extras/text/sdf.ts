import { Compute } from "../../engine";

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

const distanceShader = /* wgsl */ `
struct Uniforms {
    glyphBounds: vec4<f32>,
    maxDistance: f32,
    exponent: f32,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> segments: array<vec4<f32>>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) glyphXY: vec2<f32>,
    @location(1) @interpolate(flat) segmentIdx: u32,
}

@vertex
fn vs_distance(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) segmentIdx: u32
) -> VertexOutput {
    let uv = vec2<f32>(
        f32((vid << 1u) & 2u),
        f32(vid & 2u)
    );

    var out: VertexOutput;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.glyphXY = mix(uniforms.glyphBounds.xy, uniforms.glyphBounds.zw, uv);
    out.segmentIdx = segmentIdx;
    return out;
}

@fragment
fn fs_distance(input: VertexOutput) -> @location(0) vec4<f32> {
    let seg = segments[input.segmentIdx];
    let p = input.glyphXY;

    let lineDir = seg.zw - seg.xy;
    let lenSq = dot(lineDir, lineDir);
    let t = select(0.0, clamp(dot(p - seg.xy, lineDir) / lenSq, 0.0, 1.0), lenSq > 0.0);
    let closest = seg.xy + t * lineDir;
    let dist = distance(p, closest);

    let val = pow(1.0 - clamp(dist / uniforms.maxDistance, 0.0, 1.0), uniforms.exponent) * 0.5;

    let crosses = (seg.y > p.y) != (seg.w > p.y);
    let crossX = (seg.z - seg.x) * (p.y - seg.y) / (seg.w - seg.y) + seg.x;
    let crossingUp = crosses && (p.x < crossX) && (seg.y < seg.w);
    let crossingDown = crosses && (p.x < crossX) && (seg.y > seg.w);

    return vec4<f32>(
        select(0.0, 1.0/255.0, crossingUp),
        select(0.0, 1.0/255.0, crossingDown),
        0.0,
        val
    );
}
`;

const finalizeShader = /* wgsl */ `
@group(0) @binding(0) var intermediate: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_finalize(@builtin(vertex_index) vid: u32) -> VertexOutput {
    let uv = vec2<f32>(
        f32((vid << 1u) & 2u),
        f32(vid & 2u)
    );

    var out: VertexOutput;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = uv;
    return out;
}

@fragment
fn fs_finalize(input: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(intermediate, samp, input.uv);
    let inside = color.r != color.g;
    let val = select(color.a, 1.0 - color.a, inside);
    return vec4<f32>(val, val, val, val);
}
`;

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

    private _distancePipeline: GPURenderPipeline | null = null;
    private _finalizePipeline: GPURenderPipeline | null = null;

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
        this._exponent = config.exponent ?? 9;
        this._curveSubdivisions = config.curveSubdivisions ?? 16;

        this._sampler = this._device.createSampler({
            magFilter: "nearest",
            minFilter: "nearest",
        });
    }

    private ensurePipelines(): void {
        if (this._distancePipeline) return;

        const distanceModule = this._device.createShaderModule({ code: distanceShader });

        this._distancePipeline = this._device.createRenderPipeline({
            layout: "auto",
            vertex: {
                module: distanceModule,
                entryPoint: "vs_distance",
            },
            fragment: {
                module: distanceModule,
                entryPoint: "fs_distance",
                targets: [
                    {
                        format: "rgba8unorm",
                        blend: {
                            color: {
                                srcFactor: "one",
                                dstFactor: "one",
                                operation: "add",
                            },
                            alpha: {
                                srcFactor: "one",
                                dstFactor: "one",
                                operation: "max",
                            },
                        },
                    },
                ],
            },
            primitive: {
                topology: "triangle-list",
            },
        });

        const finalizeModule = this._device.createShaderModule({ code: finalizeShader });

        this._finalizePipeline = this._device.createRenderPipeline({
            layout: "auto",
            vertex: {
                module: finalizeModule,
                entryPoint: "vs_finalize",
            },
            fragment: {
                module: finalizeModule,
                entryPoint: "fs_finalize",
                targets: [{ format: "r8unorm" }],
            },
            primitive: {
                topology: "triangle-list",
            },
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
        this.ensurePipelines();
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

        const encoder = this._device.createCommandEncoder();
        const tempBuffers: GPUBuffer[] = [];

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

            const uniformBuffer = this._device.createBuffer({
                size: 32,
                usage: GPUBufferUsage.UNIFORM,
                mappedAtCreation: true,
            });
            new Float32Array(uniformBuffer.getMappedRange()).set([
                xMin,
                yMin,
                xMax,
                yMax,
                maxDist,
                this._exponent,
                0,
                0,
            ]);
            uniformBuffer.unmap();
            tempBuffers.push(uniformBuffer);

            const segmentBuffer = this._device.createBuffer({
                size: segments.length * 16,
                usage: GPUBufferUsage.STORAGE,
                mappedAtCreation: true,
            });
            const segmentData = new Float32Array(segmentBuffer.getMappedRange());
            for (let i = 0; i < segments.length; i++) {
                const seg = segments[i];
                segmentData[i * 4] = seg.x1;
                segmentData[i * 4 + 1] = seg.y1;
                segmentData[i * 4 + 2] = seg.x2;
                segmentData[i * 4 + 3] = seg.y2;
            }
            segmentBuffer.unmap();
            tempBuffers.push(segmentBuffer);

            const distanceBindGroup = this._device.createBindGroup({
                layout: this._distancePipeline!.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: uniformBuffer } },
                    { binding: 1, resource: { buffer: segmentBuffer } },
                ],
            });

            const distancePass = encoder.beginRenderPass({
                colorAttachments: [
                    {
                        view: this._intermediateTexture!.createView(),
                        clearValue: { r: 0, g: 0, b: 0, a: 0 },
                        loadOp: "clear",
                        storeOp: "store",
                    },
                ],
                timestampWrites: Compute.span?.("text:sdf-distance"),
            });

            distancePass.setPipeline(this._distancePipeline!);
            distancePass.setBindGroup(0, distanceBindGroup);
            distancePass.draw(3, segments.length);
            distancePass.end();

            const finalizeBindGroup = this._device.createBindGroup({
                layout: this._finalizePipeline!.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: this._intermediateTexture!.createView() },
                    { binding: 1, resource: this._sampler },
                ],
            });

            const outputView = entry.outputTexture.createView({
                baseMipLevel: 0,
                mipLevelCount: 1,
                baseArrayLayer: 0,
                arrayLayerCount: 1,
            });

            const finalizePass = encoder.beginRenderPass({
                colorAttachments: [
                    {
                        view: outputView,
                        loadOp: "load",
                        storeOp: "store",
                    },
                ],
                timestampWrites: Compute.span?.("text:sdf-finalize"),
            });

            finalizePass.setViewport(
                entry.outputX,
                entry.outputY,
                this._sdfSize,
                this._sdfSize,
                0,
                1,
            );
            finalizePass.setScissorRect(entry.outputX, entry.outputY, this._sdfSize, this._sdfSize);
            finalizePass.setPipeline(this._finalizePipeline!);
            finalizePass.setBindGroup(0, finalizeBindGroup);
            finalizePass.draw(3);
            finalizePass.end();
        }

        this._device.queue.submit([encoder.finish()]);

        for (const column of tempBuffers) column.destroy();
        this._pending = [];
    }

    destroy(): void {
        this._intermediateTexture?.destroy();
    }
}
