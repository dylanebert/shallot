import { describe, expect, test } from "bun:test";
import { Compute } from "../../engine";
import { createRegather, regatherArgsWgsl } from "./regather";

interface FakeBuffer {
    readonly label?: string;
    destroyed: boolean;
    destroy(): void;
}

function fakeDevice(): GPUDevice {
    return {
        queue: { writeBuffer() {} },
        createBuffer(descriptor: GPUBufferDescriptor) {
            return {
                label: descriptor.label,
                destroyed: false,
                destroy() {
                    this.destroyed = true;
                },
            } as FakeBuffer;
        },
        createBindGroup(descriptor: GPUBindGroupDescriptor) {
            return descriptor;
        },
    } as unknown as GPUDevice;
}

function fakePass(): GPUComputePassEncoder {
    return {
        setPipeline() {},
        setBindGroup() {},
        dispatchWorkgroups() {},
    } as unknown as GPUComputePassEncoder;
}

describe("cascade re-gather multi-source contract", () => {
    test("a max-draw preflight keeps reversed-size batches on one live output buffer", () => {
        const saved = { ...Compute };
        try {
            Object.assign(Compute, { device: fakeDevice() });
            const regather = createRegather("test");
            regather.ensure(4);
            regather.reserve(9);
            const args = regather.args() as unknown as FakeBuffer;

            regather.run(fakePass(), {} as GPUBuffer, {} as GPUBuffer, [0], [0], 0, 0);
            regather.run(
                fakePass(),
                {} as GPUBuffer,
                {} as GPUBuffer,
                [0],
                [0, 1, 2, 3, 4, 5, 6, 7, 8],
                0,
                1,
            );

            expect(regather.args()).toBe(args as unknown as GPUBuffer);
            expect(args.destroyed).toBe(false);
        } finally {
            Object.assign(Compute, saved);
        }
    });

    test("a run cannot grow the shared output after command recording starts", () => {
        const saved = { ...Compute };
        try {
            Object.assign(Compute, { device: fakeDevice() });
            const regather = createRegather("undersized-test");
            regather.ensure(4);
            regather.reserve(1);
            const args = regather.args() as unknown as FakeBuffer;

            expect(() =>
                regather.run(
                    fakePass(),
                    {} as GPUBuffer,
                    {} as GPUBuffer,
                    [0],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8],
                    0,
                ),
            ).toThrow("9 draws after a 8-draw reserve");
            expect(regather.args()).toBe(args as unknown as GPUBuffer);
            expect(args.destroyed).toBe(false);
        } finally {
            Object.assign(Compute, saved);
        }
    });

    test("a direct nonzero record preserves the source baseVertex lane", () => {
        expect(typeof regatherArgsWgsl).toBe("function");
        const wgsl = regatherArgsWgsl();
        expect(wgsl).toContain("var src = pair;");
        expect(wgsl).toContain("shadowArgs[i].baseVertex = drawArgs[src].baseVertex");
        expect(wgsl).not.toContain("shadowArgs[rec + 3u] = 0u");
    });
});
