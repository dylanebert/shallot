// Build inputs: sha256:34f01bc6763513314609bdb548bc7a13e8ce999c4463096be1ecbf3c36b58f8f
const url = new URL("shallot_audio.wasm", import.meta.url);
export default async function loadAudioWasm() {
    const response = await fetch(url);
    return response.arrayBuffer();
}
