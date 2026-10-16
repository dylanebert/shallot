// Build inputs: sha256:a1501005aeef8c40e7a9d2c08ee1f6159c4458da77491ae87811afd91be69995
const url = new URL("shallot_audio.wasm", import.meta.url);
export default async function loadAudioWasm() {
    const response = await fetch(url);
    return response.arrayBuffer();
}
