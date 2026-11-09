import { expect, test } from "bun:test";
import type { World } from "../../engine";
import { acquireCanvasTexture, type View } from "./view";

function surface() {
    const scopes: Array<GPUError | null> = [];
    const uncaptured: GPUError[] = [];
    const failure = { message: "GPUCanvasContext is not configured correctly" } as GPUError;
    let broken = true;
    let configurations = 0;
    let acquisitions = 0;
    const texture = { width: 320, height: 200 } as GPUTexture;
    const invalid = { width: 0, height: 0 } as GPUTexture;
    const validate = (error: GPUError) => {
        if (scopes.length) scopes[scopes.length - 1] = error;
        else uncaptured.push(error);
    };
    const device = {
        limits: { maxTextureDimension2D: 8192 },
        pushErrorScope(filter: GPUErrorFilter) {
            expect(filter).toBe("validation");
            scopes.push(null);
        },
        popErrorScope() {
            return Promise.resolve(scopes.pop() ?? null);
        },
    } as GPUDevice;
    const context = {
        getConfiguration() {
            return { device, format: "bgra8unorm" };
        },
        configure(config: GPUCanvasConfiguration) {
            expect(scopes).toHaveLength(0);
            expect(config.device).toBe(device);
            configurations++;
        },
        getCurrentTexture() {
            acquisitions++;
            const canvas = view.canvas!;
            if (
                canvas.width === 0 ||
                canvas.height === 0 ||
                canvas.width > 8192 ||
                canvas.height > 8192
            ) {
                validate(failure);
                return { width: canvas.width, height: canvas.height } as GPUTexture;
            }
            if (broken) validate(failure);
            return broken ? invalid : texture;
        },
    } as GPUCanvasContext;
    const view = {
        context,
        canvas: { id: "game", width: 320, height: 200 },
        width: 320,
        height: 200,
        viewportIndex: 0,
    } as View;
    const world = { gpu: { device } } as World;
    return {
        world,
        view,
        texture,
        uncaptured,
        validate,
        repair() {
            broken = false;
        },
        break() {
            broken = true;
        },
        get configurations() {
            return configurations;
        },
        get acquisitions() {
            return acquisitions;
        },
    };
}

test("a failed surface acquisition retries three reconfigures, then names the surface", async () => {
    const s = surface();
    for (let attempt = 0; attempt < 4; attempt++) {
        expect(acquireCanvasTexture(s.world, s.view)).toBeUndefined();
        await Promise.resolve();
    }
    expect(s.configurations).toBe(3);
    expect(() => acquireCanvasTexture(s.world, s.view)).toThrow(
        "Canvas surface game failed after 3 reconfigures: GPUCanvasContext is not configured correctly",
    );
    expect(s.uncaptured).toEqual([]);
});

test("a successful acquisition resets the consecutive reconfigure bound", async () => {
    const s = surface();
    for (let attempt = 0; attempt < 3; attempt++) {
        acquireCanvasTexture(s.world, s.view);
        await Promise.resolve();
    }
    s.repair();
    expect(acquireCanvasTexture(s.world, s.view)).toBe(s.texture);
    await Promise.resolve();
    s.break();
    for (let attempt = 0; attempt < 4; attempt++) {
        expect(acquireCanvasTexture(s.world, s.view)).toBeUndefined();
        await Promise.resolve();
    }
    expect(s.configurations).toBe(6);
    expect(() => acquireCanvasTexture(s.world, s.view)).toThrow("game failed after 3");
});

test("synchronous steps need not wait for acquisition validation to settle", async () => {
    const s = surface();
    s.repair();
    expect(acquireCanvasTexture(s.world, s.view)).toBe(s.texture);
    expect(acquireCanvasTexture(s.world, s.view)).toBe(s.texture);
    expect(s.acquisitions).toBe(2);
    await Promise.resolve();
    expect(acquireCanvasTexture(s.world, s.view)).toBe(s.texture);
});

test("only reconfigures spend the recovery bound, not unresolved failed acquisitions", async () => {
    const s = surface();
    for (let i = 0; i < 6; i++) acquireCanvasTexture(s.world, s.view);
    await Promise.resolve();
    for (let i = 0; i < 3; i++) {
        expect(acquireCanvasTexture(s.world, s.view)).toBeUndefined();
        await Promise.resolve();
    }
    expect(s.configurations).toBe(3);
    expect(() => acquireCanvasTexture(s.world, s.view)).toThrow("game failed after 3");
});

test("unrelated validation stays uncaptured even while acquisition validation is pending", async () => {
    const s = surface();
    acquireCanvasTexture(s.world, s.view);
    const unrelated = { message: "invalid render pipeline" } as GPUError;
    s.validate(unrelated);
    expect(s.uncaptured).toEqual([unrelated]);
    await Promise.resolve();
    s.repair();
    expect(acquireCanvasTexture(s.world, s.view)).toBe(s.texture);
    expect(s.uncaptured).toEqual([unrelated]);
});

for (const [width, height] of [
    [8193, 200],
    [320, 8193],
    [0, 200],
    [320, 0],
]) {
    test(`repairs backing ${width}x${height} before acquiring a positive-sized invalid texture`, async () => {
        const s = surface();
        s.repair();
        s.view.canvas!.width = width!;
        s.view.canvas!.height = height!;
        expect(acquireCanvasTexture(s.world, s.view)).toBe(s.texture);
        await Promise.resolve();
        expect(s.view.canvas!.width).toBe(320);
        expect(s.view.canvas!.height).toBe(200);
        expect(s.uncaptured).toEqual([]);
        expect(acquireCanvasTexture(s.world, s.view)).toBe(s.texture);
    });
}
