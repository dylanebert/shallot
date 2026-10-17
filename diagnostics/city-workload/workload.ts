import { SEED } from "../../examples/city/src/layout";

const populations = Array.from({ length: 9 }, (_, i) => 4096 * 2 ** i);

export const WORKLOAD = {
    seed: SEED,
    populations,
    kit: { meshes: 2, materials: 6 },
    width: 1280,
    height: 720,
    configurations: populations.map((population) => {
        const side = Math.ceil(Math.sqrt((population - 1) / 5));
        const extent = side * 12 + 12;
        // Odd grids put a building at zero; even grids put a street there.
        const street = side % 2 === 0 ? 0 : 6;
        return {
            population,
            poses: [
                {
                    name: "whole-city",
                    shadows: false,
                    translation: [0, extent * 1.3, extent * 1.5],
                    target: [0, 0, 0],
                },
                { name: "street", shadows: false, translation: [street, 2, 0], target: [street, 2, -100] },
                { name: "shadowed-street", shadows: true, translation: [street, 2, 0], target: [street, 2, -100] },
            ],
        };
    }),
} as const;

export function refuseReport(
    shadows: boolean,
    population: number,
    instances: number,
    labels: readonly string[],
): void {
    if (shadows) throw new Error("city workload refuses a shadow-enabled record");
    if (labels.some((label) => /shadow/i.test(label)))
        throw new Error("city workload refuses a shadow pass");
    if (instances !== population)
        throw new Error(`city workload refuses population ${population}: spawned ${instances}`);
}
