// Build inputs: sha256:7057e6ffb6d9ba9c859325bd86ba2d43a9ed5db85bb5820bc67ea75765f29bd9
const url = new URL("shallot_audio.wasm", import.meta.url);
export default async function loadAudioWasm() {
    const response = await fetch(url);
    return response.arrayBuffer();
}
