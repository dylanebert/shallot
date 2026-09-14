import { expect } from "bun:test";
import { resolve } from "node:path";
import { perspective, probeTexture } from "@dylanebert/shallot";
import { runBrowserCheck } from "@dylanebert/shallot/harness";
import { CAPTURE_CONTRACT } from "@dylanebert/shallot/harness/capture";
import { check } from "@dylanebert/shallot/harness/check";
import { frustumPlanes } from "@dylanebert/shallot/render";
import { FRUSTUM_FIXTURE, GPU_FIXTURE, GPU_SHADER } from "./fixtures/frame";

function signedDistance(planes: Float32Array, plane: number, point: readonly number[]): number {
    const base = plane * 4;
    return (
        planes[base] * point[0] +
        planes[base + 1] * point[1] +
        planes[base + 2] * point[2] +
        planes[base + 3]
    );
}

check(
    "the frustum keeps a boundary sphere visible",
    {
        claim: "the frustum keeps an exactly tangent sphere visible, so a strict boundary cull defect reds",
        subject: ["src/standard/render/frustum.ts", "src/engine/utils/math.ts"],
    },
    () => {
        const projection = perspective(
            FRUSTUM_FIXTURE.fov,
            FRUSTUM_FIXTURE.aspect,
            FRUSTUM_FIXTURE.near,
            FRUSTUM_FIXTURE.far,
        );
        const planes = frustumPlanes(projection, new Float32Array(24));
        const boundaryDistances = Array.from({ length: 6 }, (_, plane) =>
            signedDistance(planes, plane, FRUSTUM_FIXTURE.boundary),
        );
        const outsideDistances = Array.from({ length: 6 }, (_, plane) =>
            signedDistance(planes, plane, FRUSTUM_FIXTURE.outside),
        );
        const inside = (distances: readonly number[]) =>
            distances.every((distance) => distance >= -FRUSTUM_FIXTURE.radius);

        // The fixture point is exactly on the left plane. The public plane extraction and the cull
        // boundary therefore agree without inventing a tolerance or a second projection.
        expect(Math.abs(boundaryDistances[0] ?? Number.NaN)).toBeLessThan(1e-5);
        expect(inside(boundaryDistances)).toBe(true);
        expect(inside(outsideDistances)).toBe(false);
        expect(outsideDistances[0]).toBeLessThan(-FRUSTUM_FIXTURE.radius);
    },
);

check(
    "the GPU probe reads the fixed render target",
    {
        claim: "the GPU probe reads the shader-authored pixel from the fixed render target, so a clear-only or stale readback defect reds",
        size: "integration",
        requires: ["gpu"],
        subject: ["src/engine/runtime/probe.ts", "src/harness/fixtures/frame.ts"],
    },
    async () => {
        const peer = (await new Function("return import('bun-webgpu')")()) as {
            setupGlobals(): Promise<void>;
        };
        await peer.setupGlobals();
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("S1 GPU fixture refused: no real adapter");
        const device = await adapter.requestDevice();
        const target = device.createTexture({
            label: "s1-fixed-render-target",
            size: [GPU_FIXTURE.width, GPU_FIXTURE.height],
            format: GPU_FIXTURE.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        });
        try {
            const module = device.createShaderModule({ code: GPU_SHADER });
            const pipeline = device.createRenderPipeline({
                layout: "auto",
                vertex: { module, entryPoint: "vs" },
                fragment: {
                    module,
                    entryPoint: "fs",
                    targets: [{ format: GPU_FIXTURE.format }],
                },
                primitive: { topology: "triangle-list" },
            });
            const result = await probeTexture(device, target, {
                size: [GPU_FIXTURE.width, GPU_FIXTURE.height, 1],
                encode: (encoder) => {
                    const pass = encoder.beginRenderPass({
                        colorAttachments: [
                            {
                                view: target.createView(),
                                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                                loadOp: "clear",
                                storeOp: "store",
                            },
                        ],
                    });
                    pass.setPipeline(pipeline);
                    pass.draw(3);
                    pass.end();
                },
            });
            const bytes = new Uint8Array(result.bytes);
            const expected = [...GPU_FIXTURE.pixel];
            expect([result.width, result.height]).toEqual([GPU_FIXTURE.width, GPU_FIXTURE.height]);
            expect(result.bytes.byteLength).toBe(
                GPU_FIXTURE.width * GPU_FIXTURE.height * expected.length,
            );
            for (let offset = 0; offset < bytes.length; offset += expected.length) {
                expect(Array.from(bytes.slice(offset, offset + expected.length))).toEqual(expected);
            }
            return {
                ok: true,
                hardware: "real in-process WebGPU device",
                tick: 1,
            };
        } finally {
            target.destroy();
        }
    },
);

const SERVE = resolve(import.meta.dir, "fixtures/serve.ts");
const serveCommand = (extra: string[]) => (port: number) => [
    process.execPath,
    SERVE,
    "--port",
    String(port),
    ...extra,
];

check(
    "the semantic fixture reaches the final canvas",
    {
        claim: "fixture content reaches the final canvas at the fixed identity, so a blank or wrong-identity presentation reds",
        size: "integration",
        requires: ["chromium"],
        host: "mac",
        subject: [
            "src/harness/fixtures/page.ts",
            "src/harness/fixtures/scene.ts",
            "src/harness/capture.ts",
        ],
    },
    async () => {
        const verdict = await runBrowserCheck(serveCommand([]));
        const semantic = (verdict.checks ?? []).find(
            (entry) => entry.name === "tag reaches the final canvas",
        );
        expect(semantic?.ok).toBe(true);
        expect(semantic?.data?.first).toBeGreaterThan(0);
        expect(verdict.captureIdentity).toBe(
            `${CAPTURE_CONTRACT.width}x${CAPTURE_CONTRACT.height}`,
        );
        expect(verdict.reproduction.capture).toBe("final-canvas 1280x720@1 rgba8-tight");
        expect(verdict.reproduction.tick).toBe(1);
        return verdict;
    },
);
