import { timing, type Reference, type Stream } from "./comparison";

async function main() {
    const factory = (globalThis as unknown as { DracoDecoderModule: (options: { locateFile: () => string }) => Promise<Reference> }).DracoDecoderModule;
    const reference = await factory({ locateFile: () => "/reference.wasm" });
    const meta: Omit<Stream, "bytes">[] = await (await fetch("/streams")).json();
    const streams = await Promise.all(meta.map(async (stream, index) => ({ ...stream, bytes: new Uint8Array(await (await fetch(`/stream/${index}`)).arrayBuffer()) })));
    timing(reference, streams, Number(location.hash.slice(1)));
}
main().then(() => { (globalThis as unknown as { done: boolean }).done = true; }).catch((error) => {
    (globalThis as unknown as { failure: string }).failure = String(error);
});
