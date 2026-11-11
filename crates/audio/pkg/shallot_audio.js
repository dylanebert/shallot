// Build inputs: sha256:77fc11608012aa93b0a6f9b0d7e6bb35211c89315c12df5f7fbd150b4f01e6dc
const url = new URL("shallot_audio.wasm", import.meta.url);
export default async function loadAudioWasm() {
    const response = await fetch(url);
    return response.arrayBuffer();
}
