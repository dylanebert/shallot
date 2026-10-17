export const SEED = 173;
export const POPULATION = 16384;
export type Placement = {
    part: "block" | "dome";
    material: number;
    translation: [number, number, number];
    yaw: number;
    scale: [number, number, number];
};

/** Population counts reusable parts, including the street slab. */
export function layout(seed: number, population: number): Placement[] {
    if (!Number.isInteger(population) || population < 1)
        throw new Error("population must be a positive integer");
    let state = seed >>> 0;
    const random = () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return state / 4294967296;
    };
    const side = Math.ceil(Math.sqrt((population - 1) / 5));
    const result: Placement[] = [
        {
            part: "block",
            material: 0,
            translation: [0, -0.3, 0],
            yaw: 0,
            scale: [side * 12 + 12, 0.6, side * 12 + 12],
        },
    ];
    for (let building = 0; result.length < population; building++) {
        const x = ((building % side) - (side - 1) / 2) * 12;
        const z = (Math.floor(building / side) - (side - 1) / 2) * 12;
        const height = 6 + Math.floor(random() * 24);
        const width = 5 + random() * 3;
        const depth = 5 + random() * 3;
        const yaw = random() < 0.5 ? 0 : Math.PI / 2;
        const accent = 3 + Math.floor(random() * 3);
        const parts: Placement[] = [
            {
                part: "block",
                material: 1,
                translation: [x, height / 2, z],
                yaw,
                scale: [width, height, depth],
            },
            {
                part: "block",
                material: 2,
                translation: [x, height + 0.25, z],
                yaw,
                scale: [width + 0.5, 0.5, depth + 0.5],
            },
            {
                part: "block",
                material: accent,
                translation: [x, height * 0.65, z],
                yaw,
                scale: [width + 0.08, 1, depth + 0.08],
            },
            {
                part: "block",
                material: 2,
                translation: [x, 0.4, z],
                yaw,
                scale: [width + 1, 0.8, depth + 1],
            },
            {
                part: "dome",
                material: accent,
                translation: [x, height + 0.5, z],
                yaw,
                scale: [width * 0.65, 3, depth * 0.65],
            },
        ];
        result.push(...parts.slice(0, population - result.length));
    }
    return result;
}
