import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { runInNewContext } from "node:vm";
import { Listener, play, registerClip } from "../../src/core/audio";
import { createApp, Time } from "../../src/engine";
import { GlobalTransform, Transform } from "../../src/core/transform";
import {
    AudioRuntime,
    reconcileAudio,
    StandardAudioPlugin,
} from "../../src/standard/audio/runtime";
import { WORKLET_CODE } from "../../src/standard/audio/worklet";

const BLOCK = 128;
const BLOCKS_PER_FRAME = 6;
const FRAMES_PER_SECOND = 60;
const MAX_VOICES = 64;
const COMMANDS = 128;
const STRIDE = 9;
const FINISHED_OFFSET = 2 + COMMANDS * STRIDE;
const BUFFER_LENGTH = FINISHED_OFFSET + MAX_VOICES;

interface PortMessageEvent {
    data: any;
}

class AudioPort {
    onmessage: ((event: PortMessageEvent) => void) | null = null;
    onmessageerror: (() => void) | null = null;
    readonly event: PortMessageEvent = { data: undefined };

    constructor(
        private readonly owner: FakeAudioWorkletNode,
        private readonly processorSide: boolean,
    ) {}

    postMessage(message: any): void {
        if (this.processorSide) this.owner.fromProcessor(message);
        else this.owner.toProcessor(message);
    }
}

let constructingNode: FakeAudioWorkletNode;

class FakeAudioWorkletProcessor {
    readonly port: AudioPort;

    constructor() {
        this.port = constructingNode.processorPort;
    }
}

class FakeAudioWorkletNode {
    readonly processorPort = new AudioPort(this, true);
    readonly port = new AudioPort(this, false);
    readonly outputLeft = new Float32Array(BLOCK);
    readonly outputRight = new Float32Array(BLOCK);
    readonly inputs: Float32Array[][] = [[]];
    readonly outputs: Float32Array[][] = [[this.outputLeft, this.outputRight]];
    processor: any;
    pendingBatch: any = null;
    pendingRecycle: any = null;
    pendingHeartbeat: any = null;
    readonly finishedHandles = new Uint32Array(MAX_VOICES);
    pendingFinishedCount = 0;
    readonly deferredBuffer = new Float64Array(BUFFER_LENGTH);
    readonly deferredMessage = { type: "recycle", buffer: this.deferredBuffer };

    constructor(_context: FakeAudioContext, _name: string, options: AudioWorkletNodeOptions) {
        let Processor: (new (options: AudioWorkletNodeOptions) => any) | undefined;
        runInNewContext(
            WORKLET_CODE,
            {
                AudioWorkletProcessor: FakeAudioWorkletProcessor,
                registerProcessor: (_name: string, processor: unknown) => {
                    Processor = processor as new (options: AudioWorkletNodeOptions) => any;
                },
                sampleRate: 48_000,
            },
            { filename: "shallot-audio-worklet.js" },
        );
        if (!Processor) throw new Error("audio worklet processor was not registered");
        constructingNode = this;
        try {
            this.processor = new Processor(options);
        } finally {
            constructingNode = undefined as unknown as FakeAudioWorkletNode;
        }
    }

    connect(): void {}
    disconnect(): void {}

    toProcessor(message: any): void {
        if (message.type === "batch") {
            this.pendingBatch = message;
            return;
        }
        this.processorPort.event.data = message;
        this.processorPort.onmessage?.(this.processorPort.event);
    }

    processBlocks(blocks = BLOCKS_PER_FRAME): void {
        if (this.pendingBatch) {
            this.processorPort.event.data = this.pendingBatch;
            this.pendingBatch = null;
            this.processorPort.onmessage?.(this.processorPort.event);
        }
        for (let block = 0; block < blocks; block++) this.processor.process(this.inputs, this.outputs);
        if (this.pendingRecycle) {
            const message = this.pendingRecycle;
            this.pendingRecycle = null;
            this.port.event.data = message;
            this.port.onmessage?.(this.port.event);
        }
        if (this.pendingHeartbeat) {
            const message = this.pendingHeartbeat;
            this.pendingHeartbeat = null;
            this.port.event.data = message;
            this.port.onmessage?.(this.port.event);
        }
    }

    fromProcessor(message: any): void {
        if (message.type === "recycle") {
            const buffer = message.buffer as Float64Array;
            for (let i = 0; i < buffer[1]; i++) {
                const handle = buffer[FINISHED_OFFSET + i];
                let seen = false;
                for (let j = 0; j < this.pendingFinishedCount; j++) {
                    if (this.finishedHandles[j] === handle) seen = true;
                }
                if (!seen)
                    this.finishedHandles[this.pendingFinishedCount++] = handle;
            }
            buffer[1] = 0;
            this.pendingRecycle = message;
        } else if (message.type === "heartbeat") this.pendingHeartbeat = message;
        else {
            this.port.event.data = message;
            this.port.onmessage?.(this.port.event);
        }
    }
}

class FakeAudioContext {
    readonly state: AudioContextState = "running";
    readonly sampleRate = 48_000;
    readonly destination = {} as AudioDestinationNode;
    readonly audioWorklet = { addModule: async (_url: string) => {} } as AudioWorklet;

    createConstantSource(): ConstantSourceNode {
        return {
            offset: { value: 0 },
            connect() {},
            disconnect() {},
            start() {},
            stop() {},
        } as unknown as ConstantSourceNode;
    }

    addEventListener(): void {}
    removeEventListener(): void {}
    resume(): Promise<void> {
        return Promise.resolve();
    }
    close(): Promise<void> {
        return Promise.resolve();
    }
}

let controlSink: { frame: number } | undefined;
export function control(): void {
    controlSink = { frame: 0 };
}

function installAudioHost(): () => void {
    const priorContext = Object.getOwnPropertyDescriptor(globalThis, "AudioContext");
    const priorNode = Object.getOwnPropertyDescriptor(globalThis, "AudioWorkletNode");
    const priorFetch = Object.getOwnPropertyDescriptor(globalThis, "fetch");
    const fetch = globalThis.fetch;
    Object.defineProperty(globalThis, "fetch", {
        configurable: true,
        writable: true,
        value: (input: RequestInfo | URL, init?: RequestInit) => {
            const url = input instanceof URL ? input : new URL(typeof input === "string" ? input : input.url);
            if (url.protocol === "file:" && url.pathname.endsWith("shallot_audio.wasm")) {
                const root = process.env.SHALLOT_PROJECT_ROOT ?? process.cwd();
                return Promise.resolve(
                    new Response(readFileSync(resolve(root, "crates/audio/pkg/shallot_audio.wasm"))),
                );
            }
            return fetch(input, init);
        },
    });
    Object.defineProperty(globalThis, "AudioContext", {
        configurable: true,
        writable: true,
        value: FakeAudioContext,
    });
    Object.defineProperty(globalThis, "AudioWorkletNode", {
        configurable: true,
        writable: true,
        value: FakeAudioWorkletNode,
    });
    return () => {
        if (priorContext) Object.defineProperty(globalThis, "AudioContext", priorContext);
        else Reflect.deleteProperty(globalThis, "AudioContext");
        if (priorNode) Object.defineProperty(globalThis, "AudioWorkletNode", priorNode);
        else Reflect.deleteProperty(globalThis, "AudioWorkletNode");
        if (priorFetch) Object.defineProperty(globalThis, "fetch", priorFetch);
        else Reflect.deleteProperty(globalThis, "fetch");
    };
}

export default async function create() {
    const uninstall = installAudioHost();
    let app: Awaited<ReturnType<typeof createApp>> | undefined;
    try {
        app = await createApp({ defaults: false, plugins: [StandardAudioPlugin] });
        const world = app.world;
        const loopingPcm = new Float32Array(4096);
        loopingPcm.fill(0.1);
        const oncePcm = new Float32Array(4096);
        oncePcm.fill(0.1);
        const loopingClip = registerClip(world, loopingPcm);
        const onceClip = registerClip(world, oncePcm);

        const listener = world.create();
        world.add(listener, Listener);
        world.add(listener, Transform);
        const listenerTransform = world.storage(Transform);
        play(world, loopingClip, { loop: true, position: [4, 0, 0] });
        play(world, onceClip);

        const runtime = world.resource(AudioRuntime);
        const node = runtime.node as unknown as FakeAudioWorkletNode;
        if (!node) throw new Error("StandardAudioPlugin did not create its fake worklet");
        let frame = 0;
        let framesSinceWait = 0;
        function step(): void {
            listenerTransform.translation.set(listener, Math.sin(frame / 30), 0, 0, 0);
            world.step(Time.FIXED_DT);
            node.processBlocks();
            frame++;
            framesSinceWait++;
        }
        function wait(): void {
            if (framesSinceWait === 0) return;
            if (framesSinceWait !== FRAMES_PER_SECOND)
                throw new Error(`expected a one-second audio chunk, got ${framesSinceWait} frames`);
            framesSinceWait = 0;
            const count = node.pendingFinishedCount;
            if (count === 0) throw new Error("once clip did not finish within one second");
            const returned = runtime.buffer;
            if (!returned) throw new Error("audio allocation subject lost its command buffer");
            const buffer = node.deferredBuffer;
            buffer[1] = count;
            for (let i = 0; i < count; i++)
                buffer[FINISHED_OFFSET + i] = node.finishedHandles[i]!;
            node.port.event.data = node.deferredMessage;
            node.port.onmessage?.(node.port.event);
            runtime.buffer = returned;
            node.pendingFinishedCount = 0;
            play(world, onceClip);
            reconcileAudio(world);
            node.processBlocks(1);
        }
        return {
            step,
            wait,
            allocationFunctions: [AudioPort.prototype.postMessage],
            dispose() {
                app?.dispose();
                uninstall();
            },
        };
    } catch (error) {
        app?.dispose();
        uninstall();
        throw error;
    }
}
