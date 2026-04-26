<script lang="ts">
    let {
        attack,
        decay,
        sustain,
        release,
        attackCurve = 0,
        decayCurve = 0,
        releaseCurve = 0,
    }: {
        attack: number;
        decay: number;
        sustain: number;
        release: number;
        attackCurve?: number;
        decayCurve?: number;
        releaseCurve?: number;
    } = $props();

    const W = 200;
    const H = 60;
    const PAD = 6;
    const SAMPLES_PER_SEGMENT = 32;

    function curve(t: number, c: number): number {
        const p = c * 6.0;
        if (Math.abs(p) < 0.01) return t;
        return (Math.exp(p * t) - 1) / (Math.exp(p) - 1);
    }

    const path = $derived.by(() => {
        const w = W - PAD * 2;
        const h = H - PAD * 2;

        const total = attack + decay + release + 0.0001;
        const ax = (attack / total) * w;
        const dx = (decay / total) * w;
        const rx = (release / total) * w;

        const points: string[] = [];

        for (let i = 0; i <= SAMPLES_PER_SEGMENT; i++) {
            const t = i / SAMPLES_PER_SEGMENT;
            const level = curve(t, attackCurve);
            const px = PAD + t * ax;
            const py = PAD + h * (1 - level);
            points.push(`${i === 0 ? "M" : "L"}${px.toFixed(1)},${py.toFixed(1)}`);
        }

        for (let i = 1; i <= SAMPLES_PER_SEGMENT; i++) {
            const t = i / SAMPLES_PER_SEGMENT;
            const level = 1 + (sustain - 1) * curve(t, decayCurve);
            const px = PAD + ax + t * dx;
            const py = PAD + h * (1 - level);
            points.push(`L${px.toFixed(1)},${py.toFixed(1)}`);
        }

        for (let i = 1; i <= SAMPLES_PER_SEGMENT; i++) {
            const t = i / SAMPLES_PER_SEGMENT;
            const level = sustain * (1 - curve(t, releaseCurve));
            const px = PAD + ax + dx + t * rx;
            const py = PAD + h * (1 - level);
            points.push(`L${px.toFixed(1)},${py.toFixed(1)}`);
        }

        return points.join(" ");
    });
</script>

<svg class="env-viz" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
    <path d={path} fill="none" stroke="var(--accent)" stroke-width="1.5" vector-effect="non-scaling-stroke" stroke-linecap="round" stroke-linejoin="round" />
    <path d="{path} L{W - PAD},{H - PAD} L{PAD},{H - PAD} Z" fill="var(--accent)" opacity="0.08" />
</svg>

<style>
    .env-viz {
        width: 100%;
        aspect-ratio: 200 / 60;
        display: block;
    }
</style>
