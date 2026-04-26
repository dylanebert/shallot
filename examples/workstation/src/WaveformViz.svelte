<script lang="ts">
    let { waveform }: { waveform: number } = $props();

    const W = 200;
    const H = 60;
    const PAD = 6;

    const path = $derived.by(() => {
        const w = W - PAD * 2;
        const h = H - PAD * 2;
        const mid = PAD + h / 2;
        const samples = 200;
        const cycles = 1;

        const points: string[] = [];
        for (let i = 0; i < samples; i++) {
            const t = i / samples;
            const phase = t * cycles;
            const frac = phase % 1;
            let y: number;

            const RMS_SINE = Math.SQRT1_2;
            const RMS_SAW = 1 / Math.sqrt(3);
            const RMS_SQUARE = 1;
            const RMS_TRIANGLE = 1 / Math.sqrt(3);

            switch (waveform) {
                case 0: // sine
                    y = Math.sin(frac * Math.PI * 2);
                    break;
                case 1: // saw
                    y = (2 * frac - 1) * (RMS_SINE / RMS_SAW);
                    break;
                case 2: // square
                    y = (frac < 0.5 ? 1 : -1) * (RMS_SINE / RMS_SQUARE);
                    break;
                case 3: // triangle
                    y = (frac < 0.5 ? 4 * frac - 1 : 3 - 4 * frac) * (RMS_SINE / RMS_TRIANGLE);
                    break;
                default:
                    y = 0;
            }

            const px = PAD + t * w;
            const py = mid - y * (h / 2) * 0.85;
            points.push(`${i === 0 ? "M" : "L"}${px.toFixed(1)},${py.toFixed(1)}`);
        }
        return points.join(" ");
    });
</script>

<svg class="waveform-viz" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
    <path d={path} fill="none" stroke="var(--accent)" stroke-width="1.5" vector-effect="non-scaling-stroke" stroke-linecap="round" stroke-linejoin="round" />
    <path d="{path} L{W - PAD},{H / 2} L{PAD},{H / 2} Z" fill="var(--accent)" opacity="0.08" />
</svg>

<style>
    .waveform-viz {
        width: 100%;
        aspect-ratio: 200 / 60;
        display: block;
    }
</style>
