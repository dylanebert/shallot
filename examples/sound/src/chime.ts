export function chime(sampleRate: number): Float32Array {
    const pcm = new Float32Array(Math.ceil(sampleRate * 0.8));
    for (let i = 0; i < pcm.length - 1; i++) {
        const t = i / sampleRate;
        const attack = 1 - Math.exp(-t * 700);
        const release = (1 - i / (pcm.length - 1)) ** 2;
        pcm[i] =
            attack *
            release *
            (0.65 * Math.sin(2 * Math.PI * 660 * t) * Math.exp(-t * 5) +
                0.23 * Math.sin(2 * Math.PI * 1327 * t) * Math.exp(-t * 9) +
                0.12 * Math.sin(2 * Math.PI * 2648 * t) * Math.exp(-t * 16));
    }
    return pcm;
}
