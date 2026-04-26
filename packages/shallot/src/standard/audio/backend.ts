import type { ModInfo } from "./instrument";

export type AudioCommand =
    | { type: "params"; changes: [number, number, number][] }
    | { type: "gate"; voiceId: number; value: number }
    | { type: "spatial"; data: Float32Array; len: number }
    | { type: "voice_active"; voiceId: number; active: boolean }
    | {
          type: "set_instrument";
          id: number;
          nodeCount: number;
          outputBuf: number;
          nodes: {
              type: number;
              inputBuf: number;
              inputBufB: number;
              outputBuf: number;
              paramOffset: number;
          }[];
          modulations: ModInfo[];
      }
    | { type: "set_voice_instrument"; voiceId: number; instrumentId: number }
    | { type: "set_sample"; id: number; data: Float32Array }
    | { type: "watch_idle"; voiceId: number }
    | { type: "transport_play"; tid: number }
    | { type: "transport_stop"; tid: number }
    | { type: "transport_pause"; tid: number }
    | { type: "transport_set_bpm"; tid: number; bpm: number }
    | {
          type: "transport_queue_event";
          tid: number;
          beat: number;
          voiceId: number;
          durationBeats: number;
          p0Off: number;
          p0Val: number;
          p1Off: number;
          p1Val: number;
          p2Off: number;
          p2Val: number;
          p3Off: number;
          p3Val: number;
          paramCount: number;
      }
    | { type: "transport_clear_events"; tid: number }
    | { type: "transport_set_loop"; tid: number; length: number }
    | { type: "transport_seek"; tid: number; beat: number }
    | { type: "voice_spatial"; voiceId: number; spatial: boolean }
    | { type: "voice_one_shot"; voiceId: number }
    | { type: "acoustic"; data: Float32Array; len: number }
    | { type: "reflectionIR"; voiceId: number; ir: Float32Array; irLen: number }
    | { type: "reflectionGain"; voiceId: number; gain: number }
    | { type: "set_budget"; budget: number }
    | { type: "reset" }
    | {
          type: "reverb";
          rt60Low: number;
          rt60Mid: number;
          rt60High: number;
          wetGain: number;
          eqLow: number;
          eqMid: number;
          eqHigh: number;
      };

export interface AudioBackend {
    readonly running: boolean;
    init(handler: Readback): Promise<void>;
    dispose(): void;
    send(cmd: AudioCommand): void;
    pollReadback(): void;
    flush(): void;
}

export interface Readback {
    onVoiceIdle(voiceId: number): void;
    onTransportBeat(tid: number, beatLo: number, beatHi: number): void;
}
