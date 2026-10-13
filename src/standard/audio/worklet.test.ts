import { expect, test } from "bun:test";
import { runInNewContext } from "node:vm";
import { WORKLET_CODE } from "./worklet";

const FRAME = 128;
const BATCH_LENGTH = 2 + 128 * 9 + 64;

function makeProcessor() {
    const memory = new WebAssembly.Memory({ initial: 1 });
    const posted: { message: { type: string; buffer?: ArrayBuffer }; transfer?: ArrayBuffer[] }[] =
        [];
    let Processor: any;
    const exports: Record<string, unknown> = { memory };
    exports["audio_init"] = () => {};
    exports["audio_process"] = () => {
        throw new Error("worklet called the kernel's already-limited output");
    };
    exports["audio_process_pre_limiter"] = () => {
        const output = new Float32Array(memory.buffer, 0, FRAME * 2);
        output.fill(0.8, 0, FRAME);
        output.fill(0.3, FRAME);
        return 0;
    };
    exports["audio_voice_idle"] = () => 0;
    exports["audio_voice_active"] = () => {};
    class MockAudioWorkletProcessor {
        port = {
            onmessage: null as ((event: { data: unknown }) => void) | null,
            postMessage(message: { type: string; buffer?: ArrayBuffer }, transfer?: ArrayBuffer[]) {
                posted.push({ message, transfer });
            },
        };
    }
    runInNewContext(WORKLET_CODE, {
        AudioWorkletProcessor: MockAudioWorkletProcessor,
        WebAssembly: { instantiate: async () => ({ exports }) },
        registerProcessor: (_name: string, processor: unknown) => {
            Processor = processor;
        },
        sampleRate: 48000,
        Float32Array,
        Float64Array,
        Math,
    });
    const processor = new Processor({ processorOptions: { module: {} } });
    return { posted, processor };
}

test("the worklet adds its bus input before one master tanh and reuses process messages", async () => {
    const { posted, processor } = makeProcessor();
    await new Promise((resolve) => setTimeout(resolve, 0));

    const inputLeft = new Float32Array(FRAME).fill(0.4);
    const inputRight = new Float32Array(FRAME).fill(0.2);
    const left = new Float32Array(FRAME);
    const right = new Float32Array(FRAME);
    const inputs = [[inputLeft, inputRight]];
    const outputs = [[left, right]];

    for (let block = 0; block < 2; block++) {
        processor.batch = new Float64Array(BATCH_LENGTH);
        processor.batchApplied = true;
        processor.blocks = 343;
        expect(processor.process(inputs, outputs)).toBe(true);
    }

    expect(left[0]).toBeCloseTo(Math.tanh(0.8 + 0.4));
    expect(right[0]).toBeCloseTo(Math.tanh(0.3 + 0.2));
    expect(left[0]).not.toBeCloseTo(Math.tanh(Math.tanh(0.8) + 0.4));
    expect(right[0]).not.toBeCloseTo(Math.tanh(Math.tanh(0.3) + 0.2));

    const recycles = posted.filter((entry) => entry.message.type === "recycle");
    const heartbeats = posted.filter((entry) => entry.message.type === "heartbeat");
    expect(recycles).toHaveLength(2);
    expect(recycles[0].message).toBe(recycles[1].message);
    expect(recycles[0].transfer).toBe(recycles[1].transfer);
    expect(heartbeats).toHaveLength(2);
    expect(heartbeats[0].message).toBe(heartbeats[1].message);
});
