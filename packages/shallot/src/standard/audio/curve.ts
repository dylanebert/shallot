export interface CurveMapping {
    curve?: number[];
    min?: number;
    max?: number;
}

export function evalCurve(mapping: CurveMapping, value: number): number {
    const min = mapping.min ?? 0;
    const max = mapping.max ?? 1;

    if (!mapping.curve || mapping.curve.length < 4) {
        return min + value * (max - min);
    }

    const curve = mapping.curve;
    const clamped = Math.max(0, Math.min(1, value));

    if (clamped <= curve[0]) {
        return min + curve[1] * (max - min);
    }
    if (clamped >= curve[curve.length - 2]) {
        return min + curve[curve.length - 1] * (max - min);
    }

    for (let i = 0; i < curve.length - 2; i += 2) {
        const x0 = curve[i];
        const y0 = curve[i + 1];
        const x1 = curve[i + 2];
        const y1 = curve[i + 3];
        if (clamped >= x0 && clamped <= x1) {
            const t = x1 === x0 ? 0 : (clamped - x0) / (x1 - x0);
            const y = y0 + t * (y1 - y0);
            return min + y * (max - min);
        }
    }

    return min + curve[curve.length - 1] * (max - min);
}
