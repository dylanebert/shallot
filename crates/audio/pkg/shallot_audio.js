// Build inputs: sha256:a5ed72ae711285e4eb8ada4136c03de0b07ecec49a3bdcc9724bdb97b2463332
const url = new URL("shallot_audio.wasm", import.meta.url);
export default async function loadAudioWasm() {
    const response = await fetch(url);
    return response.arrayBuffer();
}
