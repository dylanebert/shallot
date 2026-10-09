import { expect, test } from "bun:test";
import { MeshPlugin } from "../../core/mesh";
import type { World } from "../../engine";
import { DEFAULT_FONT, loadFont, parseFont } from "./font";
import { isolationFont } from "./font.fixture";
import { Fonts, TextPlugin } from "./index";

test("the bundled default Inter font loads as regular text", async () => {
    const font = await loadFont(DEFAULT_FONT);
    expect(font.advance("A")).toBeGreaterThan(0);
    expect(font.advance("Ж")).toBeGreaterThan(0);
    expect(font.advance("Ω")).toBeGreaterThan(0);
});

test("TextPlugin initialization refuses a failed font load with its cause", async () => {
    const resources = new Map<object, unknown>();
    const world = {
        gpu: {
            device: {},
            root: {
                createBuffer() {
                    return {
                        $usage() {
                            return this;
                        },
                        $name() {
                            return this;
                        },
                        destroy() {},
                    };
                },
            },
        },
        resource(key: { create: (world: World) => unknown }) {
            if (!resources.has(key)) resources.set(key, key.create(world));
            return resources.get(key);
        },
    } as unknown as World;
    await MeshPlugin.initialize?.(world);
    world.resource(Fonts).register({ name: "offline", url: "https://font.invalid/offline.ttf" });

    const cause = new Error("font fetch blocked");
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (() => Promise.reject(cause)) as unknown as typeof fetch;
    try {
        const initialize = TextPlugin.initialize;
        if (!initialize) throw new Error("TextPlugin has no initializer");
        const error = await Promise.resolve(initialize(world)).then(
            () => {
                throw new Error("TextPlugin initialized with an unavailable font");
            },
            (error: unknown) => error as Error,
        );
        expect(error.message).toContain("font 0 (https://font.invalid/offline.ttf) failed to load");
        expect(error.cause).toBe(cause);
    } finally {
        globalThis.fetch = originalFetch;
    }
});

test("the generated isolation font has a nonempty rectangle glyph for every letter the GPU sweep renders", () => {
    const bytes = isolationFont();
    const font = parseFont(bytes);
    for (const char of "isolation") {
        expect(font.glyphPath(char)).toContain("L");
        expect(font.glyphBounds(char)).toEqual([0, 0, 600, 800]);
        expect(font.advance(char)).toBe(600);
    }
    let checksum = 0;
    const view = new DataView(bytes);
    for (let i = 0; i < bytes.byteLength; i += 4) checksum = (checksum + view.getUint32(i)) >>> 0;
    expect(checksum).toBe(0xb1b0afba);
});
