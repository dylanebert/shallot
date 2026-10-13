// Build inputs: sha256:b45a5bc628caba2907a5147c6a122e4885fa0ce15c44ed85f62ae7500ff44696
const url = new URL("shallot_audio.wasm", import.meta.url);
export default async function loadAudioWasm() {
    const response = await fetch(url);
    return response.arrayBuffer();
}
