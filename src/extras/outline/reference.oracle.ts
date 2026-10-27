import { test } from "bun:test";
import {
    coverage,
    masks,
    outlineReference,
    size,
    squaredDistance,
    widths,
} from "./reference.fixture";

const { render } = outlineReference(import.meta.path);

// Characterization, not an exact-distance gate. The width-bounded flood need not
// find nearest seeds outside band range; count distance errors only inside it.
for (const mask of masks)
    for (const width of widths) {
        test(`${mask.name}, width ${width}: brute-force distance characterization`, async () => {
            const actual = await render(mask.pixels, width);
            const distanceErrors: {
                pixel: number[];
                exactSquared: number;
                floodSquared: number;
            }[] = [];
            let coverageErrors = 0;
            for (let y = 0; y < size; y++)
                for (let x = 0; x < size; x++) {
                    const exactSquared = Math.min(
                        ...mask.pixels.map(([sx, sy]) => (x - sx) ** 2 + (y - sy) ** 2),
                    );
                    const floodSquared = actual.distances
                        ? actual.distances[y * size + x]! ** 2
                        : squaredDistance(actual.nearest, x, y);
                    if (exactSquared < width ** 2 && exactSquared !== floodSquared)
                        distanceErrors.push({ pixel: [x, y], exactSquared, floodSquared });
                    // Compare the distance field's coverage before the composite's output rounding.
                    if (
                        coverage(Math.sqrt(exactSquared), width) !==
                        coverage(Math.sqrt(floodSquared), width)
                    )
                        coverageErrors++;
                }
            console.info(
                JSON.stringify({
                    mask: mask.name,
                    width,
                    distanceErrorCount: distanceErrors.length,
                    coverageErrors,
                    distanceErrors,
                }),
            );
        });
    }
