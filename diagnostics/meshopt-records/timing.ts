function batch(decode: () => void, iterations: number): number {
    const start = performance.now();
    for (let i = 0; i < iterations; i++) decode();
    return performance.now() - start;
}

/** Equal work on both sides, with alternating order during warmup, calibration and sampling. */
export function timePair(own: () => void, reference: () => void, first: number): [number, number] {
    const functions = [own, reference];
    for (let i = 0; i < 1000; i++) {
        functions[(first+i)&1]();
        functions[(first+i+1)&1]();
    }
    let iterations = 1;
    for (let round = 0; ; round++) {
        const a = (first+round)&1;
        const one = batch(functions[a], iterations);
        const two = batch(functions[1-a], iterations);
        if (Math.min(one, two) >= 5) break;
        iterations *= 2;
    }
    const samples: [number[], number[]] = [[], []];
    for (let round = 0; round < 8; round++) {
        const a = (first+round)&1;
        samples[a].push(batch(functions[a], iterations)*1000/iterations);
        samples[1-a].push(batch(functions[1-a], iterations)*1000/iterations);
    }
    const median = (values: number[]) => {
        values.sort((a,b) => a-b);
        return (values[3]+values[4])/2;
    };
    return [median(samples[0]), median(samples[1])];
}
