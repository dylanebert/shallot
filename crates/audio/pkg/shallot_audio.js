// Build inputs: sha256:ce6bf8dac9d94e5a501b3e2deba605ec97530ed4007887b78ded1f5535a0a33a
const url = new URL("shallot_audio.wasm", import.meta.url);
export default async function loadAudioWasm() {
    const response = await fetch(url);
    return response.arrayBuffer();
}
