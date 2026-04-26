import type { State } from "../../engine";
import { Views } from "../../standard/viewport";

type VideoCodec = "avc" | "hevc" | "vp9" | "av1" | "vp8";

export interface Encoder {
    add(timestamp: number, duration: number): Promise<void>;
    end(): Promise<Blob>;
}

export async function encoder(
    state: State,
    config?: {
        fps?: number;
        bitrate?: number;
        codec?: VideoCodec;
        format?: "mp4" | "webm";
    },
): Promise<Encoder> {
    const views = Views.from(state);
    if (!views || views.size === 0) throw new Error("No views available");
    const canvas = views.values().next().value!.element!;

    const { Output, CanvasSource, Mp4OutputFormat, WebMOutputFormat, BufferTarget } =
        // @ts-ignore optional peer dependency
        await import("mediabunny");

    const webm = config?.format === "webm";
    const fmt = webm ? new WebMOutputFormat() : new Mp4OutputFormat({ fastStart: "in-memory" });
    const target = new BufferTarget();
    const output = new Output({ format: fmt, target });

    const source = new CanvasSource(canvas, {
        codec: config?.codec ?? "avc",
        bitrate: config?.bitrate ?? 5_000_000,
    });

    output.addVideoTrack(source, { frameRate: config?.fps ?? 60 });
    await output.start();

    return {
        add: (timestamp, duration) => source.add(timestamp, duration),
        end: async () => {
            await output.finalize();
            return new Blob([target.buffer!], { type: webm ? "video/webm" : "video/mp4" });
        },
    };
}

export async function record(
    state: State,
    duration: number,
    config?: { fps?: number; bitrate?: number; codec?: VideoCodec; format?: "mp4" | "webm" },
    onProgress?: (p: number) => void,
): Promise<Blob> {
    const fps = config?.fps ?? 60;
    const frames = Math.round(duration * fps);
    const dt = 1 / fps;

    const enc = await encoder(state, { fps, ...config });

    for (let i = 0; i < frames; i++) {
        state.step(dt);
        await enc.add(i * dt, dt);
        onProgress?.(i / frames);
    }

    return await enc.end();
}

export function download(blob: Blob, filename = "recording.mp4") {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}
