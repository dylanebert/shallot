import { afterEach, describe, expect, test } from "bun:test";
import { State } from "../../engine";
import { Compute } from "../../engine/runtime";
import { Render } from "./core";
import { RenderPlugin } from "./index";

const EndFrameSystem = RenderPlugin.systems?.find((system) => system.terminal);
if (!EndFrameSystem) throw new Error("RenderPlugin has no terminal submission system");

const mutableCompute = Compute as { device: GPUDevice | null; frame: number };
const originalDevice = Compute.device;
const originalFrame = Compute.frame;
const originalEncoder = Render.encoder;

afterEach(() => {
    mutableCompute.device = originalDevice;
    mutableCompute.frame = originalFrame;
    Render.encoder = originalEncoder;
});

describe("render submission", () => {
    test("a last producer registered after RenderPlugin encodes before the terminal submit", () => {
        const events: string[] = [];
        const encoder = {
            finish() {
                events.push("finish");
                return {} as GPUCommandBuffer;
            },
        } as GPUCommandEncoder;
        mutableCompute.frame = 0;
        mutableCompute.device = {
            queue: {
                submit() {
                    events.push("submit");
                },
            },
        } as unknown as GPUDevice;

        const state = new State();
        state.addSystem(EndFrameSystem);
        state.addSystem({
            group: "draw",
            first: true,
            update() {
                Render.encoder = encoder;
            },
        });
        state.addSystem({
            group: "draw",
            last: true,
            update() {
                if (!Render.encoder) throw new Error("producer ran after submission");
                events.push("encode");
            },
        });

        state.step();

        expect(events).toEqual(["encode", "finish", "submit"]);
        expect(Compute.frame).toBeGreaterThan(0);
        expect(Render.encoder).toBeNull();
    });

    test("a present device without a frame-open encoder names BeginFrameSystem", () => {
        mutableCompute.device = { queue: { submit() {} } } as unknown as GPUDevice;
        Render.encoder = null;

        expect(() => EndFrameSystem.update?.(new State())).toThrow(/BeginFrameSystem/);
    });

    test("headless or device-loss submission remains a no-op", () => {
        mutableCompute.device = null;
        Render.encoder = null;

        expect(() => EndFrameSystem.update?.(new State())).not.toThrow();
    });
});
