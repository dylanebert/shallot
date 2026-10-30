// Build inputs: sha256:5423c643bd5e122633dea8152b15dfddd57e1af701e71e2607069733e03cec2f
const url = new URL("shallot_audio.wasm", import.meta.url);
export default async function loadAudioWasm() {
    const response = await fetch(url);
    return response.arrayBuffer();
}
