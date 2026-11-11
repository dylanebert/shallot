import { attribution } from "./attribution";
import { timing, type Input, type Reference } from "./comparison";

async function main() {
    const factory = (globalThis as unknown as { BASIS: (options: { locateFile: () => string }) => Promise<Reference & { initializeBasis(): void }> }).BASIS;
    const reference = await factory({ locateFile: () => "/reference.wasm" });
    reference.initializeBasis();
    const paths: string[] = await (await fetch("/inputs")).json();
    const attribute = new URL(location.href).searchParams.has("attribution");
    const files: Input[] = await Promise.all(paths.map(async (path) => {
        const bytes = new Uint8Array(await (await fetch(`/input/${path}`)).arrayBuffer());
        const plain = attribute ? new Uint8Array(await (await fetch(`/plain/${path}`)).arrayBuffer()) : undefined;
        return { path, bytes, plain };
    }));
    if (attribute) attribution(reference, files);
    else timing(reference, files, Number(location.hash.slice(1)));
}
main().then(() => { (globalThis as unknown as { done: boolean }).done = true; }).catch((error) => {
    (globalThis as unknown as { failure: string }).failure = String(error);
});
