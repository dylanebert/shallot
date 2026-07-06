// The decode worker — a thin shell: receive (url, clip, targets), run the SAME deviceless `decode`, and post
// the payload back with its typed arrays + bitmaps as transferables (zero-copy). One decode implementation,
// two call sites (inline for tests, this for runtime) — the worker is transport, never a second decode. Loaded
// as a module worker by `decodeInWorker` (worker.ts); the Draco / Basis wasm it pulls in resolve via the
// codecs' own `import.meta.url`, so they bundle into the worker chunk the downstream app builds.
import { decode } from "./assets";
import { type DecodeReply, type DecodeRequest, transferables } from "./worker";

// the worker global, typed without the WebWorker lib (tsconfig ships DOM only): just the two members this
// shell touches. In a dedicated worker `globalThis` is the worker scope, so `onmessage` / `postMessage` wire
// the message channel.
const ctx = globalThis as unknown as {
    onmessage: ((e: MessageEvent<DecodeRequest>) => void) | null;
    postMessage(message: DecodeReply, transfer?: Transferable[]): void;
};

ctx.onmessage = async (e) => {
    const { url, clip, targets } = e.data;
    try {
        const decoded = await decode(url, { clip, targets });
        ctx.postMessage({ ok: true, decoded }, transferables(decoded));
    } catch (err) {
        // forward the message only — an Error doesn't survive the structured clone, and the pool rewraps this
        // in an Error main-thread-side. warnUnsupported (an intentional-unsupported feature) never reaches here:
        // it console.warns inside decode without throwing, so it forwards to devtools from this worker context.
        ctx.postMessage({ ok: false, error: err instanceof Error ? err.message : String(err) });
    }
};
