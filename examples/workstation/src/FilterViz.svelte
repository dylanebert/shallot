<script lang="ts">
    import { resToQ } from "./presets";

    let {
        mode,
        cutoff,
        res,
        mix,
    }: {
        mode: number;
        cutoff: number;
        res: number;
        mix: number;
    } = $props();

    const W = 200;
    const H = 60;
    const PAD = 6;

    function svfMagnitudeDb(freq: number, fc: number, qVal: number, filterMode: number): number {
        const s = freq / fc;
        const s2 = s * s;
        const k = 1 / qVal;
        const denom = (1 - s2) * (1 - s2) + k * k * s2;

        let magSq: number;
        switch (filterMode) {
            case 0: // LP: H = 1 / (1 + ks + s^2)
                magSq = 1 / denom;
                break;
            case 1: // HP: H = s^2 / (1 + ks + s^2)
                magSq = s2 * s2 / denom;
                break;
            case 2: // BP: H = ks / (1 + ks + s^2)
                magSq = k * k * s2 / denom;
                break;
            case 3: // Notch: H(s) = (s^2 + 1) / (s^2 + ks + 1)
                magSq = ((1 - s2) * (1 - s2)) / denom;
                break;
            default:
                magSq = 1;
        }

        return 10 * Math.log10(Math.max(magSq, 1e-10));
    }

    const path = $derived.by(() => {
        const w = W - PAD * 2;
        const h = H - PAD * 2;
        const samples = 200;

        const logMin = Math.log10(20);
        const logMax = Math.log10(40000);

        const points: string[] = [];
        for (let i = 0; i <= samples; i++) {
            const t = i / samples;
            const freq = 10 ** (logMin + t * (logMax - logMin));
            const q = resToQ(res);
            const rawDb = mode === 0 ? 0 : svfMagnitudeDb(freq, cutoff, q, mode - 1);
            const wet = 10 ** (rawDb / 20);
            let db = 20 * Math.log10((1 - mix) + mix * wet);
            db = Math.max(-36, Math.min(24, db));
            const norm = (db + 36) / 60;
            const px = PAD + t * w;
            const py = PAD + h * (1 - norm);
            points.push(`${i === 0 ? "M" : "L"}${px.toFixed(1)},${py.toFixed(1)}`);
        }
        return points.join(" ");
    });
</script>

<svg class="filter-viz" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
    <path d={path} fill="none" stroke="var(--accent)" stroke-width="1.5" vector-effect="non-scaling-stroke" stroke-linecap="round" stroke-linejoin="round" />
    <path d="{path} L{W - PAD},{H - PAD} L{PAD},{H - PAD} Z" fill="var(--accent)" opacity="0.08" />
</svg>

<style>
    .filter-viz {
        width: 100%;
        aspect-ratio: 200 / 60;
        display: block;
    }
</style>
