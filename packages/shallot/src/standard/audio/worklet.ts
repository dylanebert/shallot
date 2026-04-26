const MAX_TRANSPORTS = 8;

const WORKLET_CODE = `
const MAX_TRANSPORTS = ${MAX_TRANSPORTS};

class SynthProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.wasm = null;
        this.memory = null;
        this._pendingMessages = [];
        this._releasing = new Set();
        this._frameCount = 0;
        this._outputPeak = 0;
        this._lastFrame = -1;
        this._droppedBlocks = 0;
        this._warmedUp = false;
        this.port.onmessage = (e) => {
            if (e.data.type === "init") {
                WebAssembly.instantiate(e.data.bytes, {}).then((result) => {
                    this.wasm = result.instance.exports;
                    this.memory = this.wasm.memory;
                    this.wasm.audio_init(sampleRate);
                    for (const msg of this._pendingMessages) {
                        this._handleMessage(msg);
                    }
                    this._pendingMessages = [];
                });
            } else {
                if (!this.wasm) {
                    this._pendingMessages.push(e.data);
                } else {
                    this._handleMessage(e.data);
                }
            }
        };
    }

    _handleMessage(data) {
        if (data.type === "batch") {
            for (const cmd of data.commands) this._handleMessage(cmd);
            return;
        }
        if (data.type === "params") {
            for (const [voiceId, paramId, value] of data.changes) {
                this.wasm.audio_set_param(voiceId, paramId, value);
            }
        } else if (data.type === "gate") {
            this.wasm.audio_set_gate(data.voiceId, data.value);
            if (data.value !== 0) {
                this._releasing.delete(data.voiceId);
            }
        } else if (data.type === "spatial") {
            const arr = data.data;
            for (let i = 0; i < arr.length; i += 7) {
                this.wasm.audio_set_spatial(
                    arr[i],
                    arr[i + 1],
                    arr[i + 2],
                    arr[i + 3],
                    arr[i + 4],
                    arr[i + 5],
                    arr[i + 6],
                );
            }
        } else if (data.type === "voice_active") {
            this.wasm.audio_voice_active(data.voiceId, data.active ? 1 : 0);
            if (!data.active) {
                this._releasing.delete(data.voiceId);
            }
        } else if (data.type === "set_instrument") {
            this.wasm.audio_set_instrument(data.id, data.nodeCount, data.outputBuf);
            for (let i = 0; i < data.nodes.length; i++) {
                const n = data.nodes[i];
                this.wasm.audio_set_instrument_node(data.id, i, n.type, n.inputBuf, n.inputBufB, n.outputBuf, n.paramOffset);
            }
            if (data.modulations) {
                for (let i = 0; i < data.modulations.length; i++) {
                    const m = data.modulations[i];
                    this.wasm.audio_set_instrument_mod(data.id, i, m.sourceBuf, m.targetNode, m.targetParam, m.depthParam, m.mode);
                }
            }
        } else if (data.type === "set_voice_instrument") {
            this.wasm.audio_set_voice_instrument(data.voiceId, data.instrumentId);
            this._releasing.delete(data.voiceId);
        } else if (data.type === "transport_play") {
            this.wasm.transport_play(data.tid);
        } else if (data.type === "transport_stop") {
            this.wasm.transport_stop(data.tid);
        } else if (data.type === "transport_pause") {
            this.wasm.transport_pause(data.tid);
        } else if (data.type === "transport_set_bpm") {
            this.wasm.transport_set_bpm(data.tid, data.bpm);
        } else if (data.type === "transport_queue_event") {
            this.wasm.transport_queue_event(
                data.tid,
                data.beat, data.voiceId, data.durationBeats,
                data.p0Off ?? 0, data.p0Val ?? 0,
                data.p1Off ?? 0, data.p1Val ?? 0,
                data.p2Off ?? 0, data.p2Val ?? 0,
                data.p3Off ?? 0, data.p3Val ?? 0,
                data.paramCount ?? 0,
            );
        } else if (data.type === "transport_clear_events") {
            this.wasm.transport_clear_events(data.tid);
        } else if (data.type === "transport_set_loop") {
            this.wasm.transport_set_loop(data.tid, data.length);
        } else if (data.type === "transport_seek") {
            this.wasm.transport_seek(data.tid, data.beat);
        } else if (data.type === "voice_spatial") {
            this.wasm.audio_set_voice_spatial(data.voiceId, data.spatial ? 1 : 0);
        } else if (data.type === "voice_one_shot") {
            this.wasm.audio_set_voice_one_shot(data.voiceId, 1);
        } else if (data.type === "acoustic") {
            const arr = data.data;
            for (let i = 0; i < data.len; i += 5) {
                this.wasm.audio_set_acoustic_separate(arr[i], arr[i + 1], arr[i + 2], arr[i + 3], arr[i + 4]);
            }
        } else if (data.type === "set_sample") {
            const ptr = this.wasm.audio_sample_alloc(data.id, data.data.length);
            if (ptr) {
                new Float32Array(this.memory.buffer, ptr, data.data.length).set(data.data);
            }
        } else if (data.type === "reflectionIR") {
            const ptr = this.wasm.audio_ir_staging_ptr();
            const staging = new Float32Array(this.memory.buffer, ptr, data.irLen);
            staging.set(data.ir.subarray(0, data.irLen));
            this.wasm.audio_set_reflection_ir(data.voiceId, data.irLen);
        } else if (data.type === "reflectionGain") {
            this.wasm.audio_set_reflection_gain(data.voiceId, data.gain);
        } else if (data.type === "reverb") {
            this.wasm.audio_set_reverb(data.rt60Low, data.rt60Mid, data.rt60High, data.wetGain, data.eqLow, data.eqMid, data.eqHigh);
        } else if (data.type === "watch_idle") {
            this._releasing.add(data.voiceId);
        } else if (data.type === "reset") {
            this.wasm.audio_reset();
            this._lastFrame = -1;
            this._warmedUp = false;
        } else if (data.type === "set_budget") {
            this.wasm.audio_set_real_voice_budget(data.budget);
        }
    }

    process(inputs, outputs, parameters) {
        if (!this.wasm) return true;
        try {
        if (this._lastFrame >= 0) {
            const gap = currentFrame - this._lastFrame;
            if (gap > 128) {
                this._droppedBlocks += (gap / 128) - 1;
            }
        }
        this._lastFrame = currentFrame;
        const ptr = this.wasm.audio_process();
        const buf = this.memory.buffer;
        const stereo = new Float32Array(buf, ptr, 256);
        const out = outputs[0];
        if (out.length >= 2 && out[0].length >= 128) {
            out[0].set(stereo.subarray(0, 128));
            out[1].set(stereo.subarray(128, 256));
        } else if (out.length === 1 && out[0].length >= 128) {
            for (let i = 0; i < 128; i++) {
                out[0][i] = (stereo[i] + stereo[128 + i]) * 0.5;
            }
        }

        const rb = new Uint32Array(buf, this.wasm.transport_readback_ptr(), MAX_TRANSPORTS * 4);
        const beats = [];
        for (let tid = 0; tid < MAX_TRANSPORTS; tid++) {
            const base = tid * 4;
            if (rb[base] !== 0) {
                beats.push({ tid, beatLo: rb[base + 1], beatHi: rb[base + 2] });
            }
        }
        if (beats.length > 0) {
            this.port.postMessage({ type: "transport_beats", beats });
        }

        const idle = [];
        for (const voiceId of this._releasing) {
            if (this.wasm.audio_voice_idle(voiceId)) {
                idle.push(voiceId);
            }
        }
        for (const voiceId of idle) {
            this.port.postMessage({ type: "voice_idle", voiceId });
            this._releasing.delete(voiceId);
        }

        const overflow = this.wasm.audio_overflow_count();
        if (overflow > 0) {
            this.port.postMessage({ type: "overflow", count: overflow });
        }

        let outPeak = 0;
        let hasNaN = false;
        if (out.length >= 2) {
            for (let i = 0; i < 128; i++) {
                const l = out[0][i];
                const r = out[1][i];
                if (l !== l || r !== r) { hasNaN = true; break; }
                const a = Math.abs(l);
                if (a > outPeak) outPeak = a;
                const b = Math.abs(r);
                if (b > outPeak) outPeak = b;
            }
        }
        if (hasNaN) outPeak = -1;
        if (outPeak > this._outputPeak || (outPeak < 0 && this._outputPeak >= 0)) this._outputPeak = outPeak;

        if (++this._frameCount % 344 === 0) {
            if (!this._warmedUp) {
                this._warmedUp = true;
                this._droppedBlocks = 0;
            }
            this.port.postMessage({
                type: "heartbeat",
                frame: this._frameCount,
                outputPeak: this._outputPeak,
                dropped: this._droppedBlocks,
            });
            this._outputPeak = 0;
            this._droppedBlocks = 0;
        }
        } catch (e) {
            this.port.postMessage({ type: "error", message: String(e) });
        }
        return true;
    }
}

registerProcessor("synth-processor", SynthProcessor);
`;

export function createWorkletURL(): string {
    const blob = new Blob([WORKLET_CODE], { type: "application/javascript" });
    return URL.createObjectURL(blob);
}
