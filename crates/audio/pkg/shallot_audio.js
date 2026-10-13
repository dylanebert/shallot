// Build inputs: sha256:f6bd9e5ef79a40f7ad2016926248c60a9639a4edaee5f644097d24b35072e9a2
const url = new URL("shallot_audio.wasm", import.meta.url);
export default async function loadAudioWasm() {
    const response = await fetch(url);
    return response.arrayBuffer();
}
