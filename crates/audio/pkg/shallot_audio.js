// Build inputs: sha256:dcddbe344f043b7125a35727aecc4b346f4f725054768b6276cda2aa3c803ab7
const url = new URL("shallot_audio.wasm", import.meta.url);
export default async function loadAudioWasm() {
    const response = await fetch(url);
    return response.arrayBuffer();
}
