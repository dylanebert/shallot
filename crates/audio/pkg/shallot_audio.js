// Build inputs: sha256:f279a6e456a6ca7d5f22fa2f71fb5e9f90117b462a94e677d5705ff5899bb505
const url = new URL("shallot_audio.wasm", import.meta.url);
export default async function loadAudioWasm() {
    const response = await fetch(url);
    return response.arrayBuffer();
}
