// Build inputs: sha256:447ffcf35a73ce6a5569dbd36a8b00831b581810ca2e7fc9c46d1a2f37080fe1
const url = new URL("shallot_audio.wasm", import.meta.url);
export default async function loadAudioWasm() {
    const response = await fetch(url);
    return response.arrayBuffer();
}
