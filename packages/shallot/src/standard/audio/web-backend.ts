import type { AudioBackend, AudioCommand, Readback } from "./backend";
import { createWorkletURL } from "./worklet";
import loadAudioWasm from "../../../rust/audio/pkg/shallot_audio.js";

export class WebBackend implements AudioBackend {
    private _audioCtx: AudioContext | null = null;
    private _workletNode: AudioWorkletNode | null = null;
    private _resumeListener: (() => void) | null = null;
    private _stateListener: (() => void) | null = null;
    private _visibilityListener: (() => void) | null = null;
    private _deviceListener: (() => void) | null = null;
    private _wasSuspended = false;
    private _queue: any[] = [];
    private _irPool: Float32Array[] = [];
    private _lastHeartbeat = 0;
    private _heartbeatTimer: ReturnType<typeof setInterval> | null = null;

    get running(): boolean {
        return this._audioCtx !== null && this._audioCtx.state === "running";
    }

    private reconnect(): void {
        if (!this._workletNode || !this._audioCtx) return;
        this._workletNode.disconnect();
        this._workletNode.connect(this._audioCtx.destination);
    }

    async init(handler: Readback): Promise<void> {
        this._audioCtx = new AudioContext();
        if (this._audioCtx.state === "suspended") {
            const resume = () => {
                this._audioCtx!.resume();
                document.removeEventListener("pointerdown", resume);
                document.removeEventListener("keydown", resume);
                this._resumeListener = null;
            };
            document.addEventListener("pointerdown", resume);
            document.addEventListener("keydown", resume);
            this._resumeListener = resume;
        }

        const wasmBytes = await loadAudioWasm();

        const workletURL = createWorkletURL();
        await this._audioCtx.audioWorklet.addModule(workletURL);
        URL.revokeObjectURL(workletURL);

        this._workletNode = new AudioWorkletNode(this._audioCtx, "synth-processor", {
            outputChannelCount: [2],
        });
        this._workletNode.connect(this._audioCtx.destination);
        this._workletNode.port.postMessage({ type: "init", bytes: wasmBytes });

        this._wasSuspended = this._audioCtx.state !== "running";
        this._stateListener = () => {
            if (this._audioCtx!.state === "running" && this._wasSuspended) {
                this._workletNode?.port.postMessage({ type: "reset" });
                this.reconnect();
            }
            this._wasSuspended = this._audioCtx!.state !== "running";
        };
        this._audioCtx.addEventListener("statechange", this._stateListener);

        this._visibilityListener = () => {
            if (document.visibilityState === "visible" && this._audioCtx) {
                this._audioCtx.resume();
                this.reconnect();
            }
        };
        document.addEventListener("visibilitychange", this._visibilityListener);

        this._deviceListener = () => this.reconnect();
        navigator.mediaDevices?.addEventListener("devicechange", this._deviceListener);

        this._workletNode.onprocessorerror = (e) => {
            console.error("audio worklet crashed:", e);
        };
        this._workletNode.port.onmessage = (e: MessageEvent) => {
            if (e.data.type === "voice_idle") {
                handler.onVoiceIdle(e.data.voiceId);
            } else if (e.data.type === "transport_beats") {
                for (const entry of e.data.beats) {
                    handler.onTransportBeat(entry.tid, entry.beatLo, entry.beatHi);
                }
            } else if (e.data.type === "overflow") {
                console.warn(`audio: ${e.data.count} events dropped (transport buffer full)`);
            } else if (e.data.type === "heartbeat") {
                this._lastHeartbeat = performance.now();
                const d = e.data;
                if (d.outputPeak !== undefined && d.outputPeak < 0) {
                    console.error("audio: NaN detected in output");
                }
                if (d.dropped > 0) {
                    console.error(`audio: ${d.dropped} blocks dropped`);
                }
            } else if (e.data.type === "error") {
                console.error(`audio worklet error: ${e.data.message}`);
            }
        };

        this._lastHeartbeat = performance.now();
        this._heartbeatTimer = setInterval(() => {
            if (!this._audioCtx || this._audioCtx.state !== "running") return;
            const elapsed = performance.now() - this._lastHeartbeat;
            if (elapsed > 3000) {
                console.warn(
                    `audio: worklet heartbeat lost (${(elapsed / 1000).toFixed(0)}s), reconnecting`,
                );
                this.reconnect();
                this._lastHeartbeat = performance.now();
            }
        }, 2000);
    }

    dispose(): void {
        this.flush();
        if (this._heartbeatTimer) {
            clearInterval(this._heartbeatTimer);
            this._heartbeatTimer = null;
        }
        if (this._resumeListener) {
            document.removeEventListener("pointerdown", this._resumeListener);
            document.removeEventListener("keydown", this._resumeListener);
            this._resumeListener = null;
        }
        if (this._stateListener && this._audioCtx) {
            this._audioCtx.removeEventListener("statechange", this._stateListener);
            this._stateListener = null;
        }
        if (this._visibilityListener) {
            document.removeEventListener("visibilitychange", this._visibilityListener);
            this._visibilityListener = null;
        }
        if (this._deviceListener) {
            navigator.mediaDevices?.removeEventListener("devicechange", this._deviceListener);
            this._deviceListener = null;
        }
        if (this._workletNode) {
            this._workletNode.disconnect();
            this._workletNode = null;
        }
        if (this._audioCtx) {
            this._audioCtx.close();
            this._audioCtx = null;
        }
    }

    send(cmd: AudioCommand): void {
        if (!this._workletNode) return;
        if (cmd.type === "params" && cmd.changes.length === 0) return;
        if (cmd.type === "spatial") {
            if (cmd.len === 0) return;
            this._queue.push({ type: "spatial", data: cmd.data.slice(0, cmd.len) });
            return;
        }
        if (cmd.type === "reflectionIR") {
            let buf = this._irPool.pop();
            if (!buf || buf.length < cmd.irLen) buf = new Float32Array(cmd.irLen);
            buf.set(cmd.ir.subarray(0, cmd.irLen));
            this._queue.push({
                type: "reflectionIR",
                voiceId: cmd.voiceId,
                ir: buf,
                irLen: cmd.irLen,
            });
            return;
        }
        this._queue.push(cmd);
    }

    pollReadback(): void {}

    flush(): void {
        if (!this._workletNode || this._queue.length === 0) return;
        for (let i = 0; i < this._queue.length; i++) {
            if (this._queue[i].type === "reflectionIR") this._irPool.push(this._queue[i].ir);
        }
        this._workletNode.port.postMessage({ type: "batch", commands: this._queue });
        this._queue.length = 0;
    }
}
