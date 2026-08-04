export interface RenderedClassification {
    rendered: boolean;
    center: number[];
    corner: number[];
}

const SIZE = 64;
const spread = (a: number[], b: number[]): number =>
    Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) + Math.abs(a[2] - b[2]);

/** Classify one 64×64 RGB screenshot grid as rendered. */
export function classifyRendered(grid: number[]): RenderedClassification {
    if (grid.length !== SIZE * SIZE * 3)
        throw new Error(`visualization gate: expected ${SIZE}×${SIZE} RGB grid`);
    const region = (fx0: number, fy0: number, fx1: number, fy1: number): number[] => {
        const x0 = Math.floor(fx0 * SIZE);
        const x1 = Math.max(x0 + 1, Math.floor(fx1 * SIZE));
        const y0 = Math.floor(fy0 * SIZE);
        const y1 = Math.max(y0 + 1, Math.floor(fy1 * SIZE));
        let r = 0;
        let g = 0;
        let b = 0;
        let n = 0;
        for (let y = y0; y < y1; y++)
            for (let x = x0; x < x1; x++) {
                const at = (y * SIZE + x) * 3;
                r += grid[at];
                g += grid[at + 1];
                b += grid[at + 2];
                n++;
            }
        return [r / n, g / n, b / n];
    };
    const center = region(0.4, 0.4, 0.6, 0.6);
    const corner = region(0, 0, 0.15, 0.15);
    return { rendered: spread(center, corner) > 12, center, corner };
}
