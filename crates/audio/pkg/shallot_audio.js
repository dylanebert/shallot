// Build inputs: sha256:3dbf6b5650e0e0d583c8cd5b11129dc78a96e976d2de0cdaba7f802ba3dbd406
const url = new URL("shallot_audio.wasm", import.meta.url);
export default async function loadAudioWasm() {
    const response = await fetch(url);
    return response.arrayBuffer();
}
