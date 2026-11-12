// Build inputs: sha256:df4fa2f0f106ac1f6f457d2df8d879902a7e559cd9ae09a008748af1b74a581f
const url = new URL("shallot_audio.wasm", import.meta.url);
export default async function loadAudioWasm() {
    const response = await fetch(url);
    return response.arrayBuffer();
}
