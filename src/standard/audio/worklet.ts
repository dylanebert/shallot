const MAX_VOICES = 64;
const COMMANDS = 128;
const STRIDE = 9;
const FINISHED_OFFSET = 2 + COMMANDS * STRIDE;
const PLAY = 1;
const UPDATE = 2;
const STOP = 3;

export const WORKLET_CODE = `
const MAX_VOICES = ${MAX_VOICES};
const COMMANDS = ${COMMANDS};
const STRIDE = ${STRIDE};
const FINISHED_OFFSET = ${FINISHED_OFFSET};
const PLAY = ${PLAY};
const UPDATE = ${UPDATE};
const STOP = ${STOP};

class ShallotAudioProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        this.wasm = null;
        this.memory = null;
        this.memoryBuffer = null;
        this.outputPtr = 0;
        this.viewPtr = -1;
        this.output = null;
        this.watched = new Float64Array(MAX_VOICES);
        this.batch = null;
        this.batchApplied = false;
        this.blocks = 0;
        this.recycleMessage = { type: "recycle", buffer: null };
        this.recycleTransfer = [null];
        this.heartbeatMessage = { type: "heartbeat" };
        this.errorMessage = { type: "error", message: "" };
        this.port.onmessage = (event) => {
            const message = event.data;
            if (message.type === "clip") {
                if (!this.wasm) return;
                for (let channel = 0; channel < message.channels.length; channel++) {
                    const pcm = message.channels[channel];
                    const ptr = this.wasm.audio_sample_alloc(message.id, channel, message.channels.length, pcm.length);
                    if (ptr) {
                        this.refreshMemory();
                        new Float32Array(this.memory.buffer, ptr, pcm.length).set(pcm);
                    }
                }
            } else if (message.type === "batch") {
                this.batch = new Float64Array(message.buffer);
                this.batchApplied = false;
                if (this.wasm) this.applyBatch();
            } else if (message.type === "reset" && this.wasm) {
                this.wasm.audio_reset();
                this.watched.fill(0);
            }
        };
        WebAssembly.instantiate(options.processorOptions.module, {}).then((instance) => {
            this.wasm = instance.exports;
            this.memory = this.wasm.memory;
            this.wasm.audio_init(sampleRate);
            this.refreshMemory();
            this.port.postMessage({ type: "ready" });
        }).catch((error) => {
            this.port.postMessage({ type: "init-error", message: String(error) });
        });
    }

    refreshMemory() {
        if (this.memoryBuffer === this.memory.buffer && this.viewPtr === this.outputPtr) return;
        this.memoryBuffer = this.memory.buffer;
        this.viewPtr = this.outputPtr;
        this.output = new Float32Array(this.memoryBuffer, this.outputPtr, 256);
    }

    applyBatch() {
        const buffer = this.batch;
        if (!buffer) return;
        const count = buffer[0];
        for (let i = 0; i < count; i++) {
            const offset = 2 + i * STRIDE;
            const op = buffer[offset];
            const handle = buffer[offset + 1];
            const voice = handle % MAX_VOICES;
            if (op === PLAY) {
                const clip = buffer[offset + 2];
                const loop = buffer[offset + 3];
                const gain = buffer[offset + 4];
                this.wasm.audio_play_clip(voice, clip, loop, gain);
                this.setSpatial(buffer, offset, voice);
                this.watched[voice] = loop === 0 ? handle : 0;
            } else if (op === UPDATE) {
                const loop = buffer[offset + 3];
                this.wasm.audio_set_clip_gain(voice, buffer[offset + 4]);
                this.wasm.audio_set_clip_loop(voice, loop);
                this.setSpatial(buffer, offset, voice);
                this.watched[voice] = loop === 0 ? handle : 0;
            } else if (op === STOP) {
                this.wasm.audio_stop_clip(voice);
                this.watched[voice] = handle;
            }
        }
        this.batchApplied = true;
    }

    setSpatial(buffer, offset, voice) {
        if (buffer[offset + 5] !== 0) {
            this.wasm.audio_set_voice_spatial(voice, 1);
            this.wasm.audio_set_spatial(voice, buffer[offset + 6], buffer[offset + 7], buffer[offset + 8], 3, 100, 1);
        } else {
            this.wasm.audio_set_voice_spatial(voice, 0);
        }
    }

    process(inputs, outputs) {
        if (!this.wasm) return true;
        try {
            if (!this.batchApplied) this.applyBatch();
            const ptr = this.wasm.audio_process_pre_limiter();
            this.outputPtr = ptr;
            this.refreshMemory();
            const output = outputs[0];
            const input = inputs[0];
            const left = output[0];
            const right = output[1];
            const inputLeft = input && input.length ? input[0] : null;
            const inputRight = input && input.length > 1 ? input[1] : inputLeft;
            if (left && right) {
                for (let i = 0; i < left.length; i++) {
                    const l = this.output[i] + (inputLeft ? inputLeft[i] : 0);
                    const r = this.output[128 + i] + (inputRight ? inputRight[i] : 0);
                    left[i] = Math.tanh(l);
                    right[i] = Math.tanh(r);
                }
            }
            let finished = 0;
            if (this.batch) {
                for (let voice = 0; voice < MAX_VOICES; voice++) {
                    const handle = this.watched[voice];
                    if (handle !== 0 && this.wasm.audio_voice_idle(voice)) {
                        this.wasm.audio_voice_active(voice, 0);
                        this.watched[voice] = 0;
                        this.batch[FINISHED_OFFSET + finished++] = handle;
                    }
                }
                this.batch[1] = finished;
                const buffer = this.batch.buffer;
                this.recycleMessage.buffer = buffer;
                this.recycleTransfer[0] = buffer;
                this.port.postMessage(this.recycleMessage, this.recycleTransfer);
                this.batch = null;
                this.batchApplied = false;
            }
            if (++this.blocks % 344 === 0) this.port.postMessage(this.heartbeatMessage);
        } catch (error) {
            this.errorMessage.message = String(error);
            this.port.postMessage(this.errorMessage);
        }
        return true;
    }
}

registerProcessor("shallot-audio", ShallotAudioProcessor);
`;

export function createWorkletURL(): string {
    return URL.createObjectURL(new Blob([WORKLET_CODE], { type: "application/javascript" }));
}
