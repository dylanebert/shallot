/** Propagate the measured L2 norm of all phase-trig lane errors through the block-diagonal
 * spectrum update. For one mode, the two rows mapping `(cos, sin, cos(-), sin(-))` error to the
 * complex height error are orthogonal and each has squared norm `|h0(k)|² + |h0(-k)|²`; slope
 * multiplication scales that operator norm by `|kx|` or `|kz|`. The whole field's spectral
 * operator norm is therefore the largest mode norm; the unnormalized 2D inverse DFT then has L2
 * operator norm N. No phase maximum, fitted margin, or channel scale enters the result. */
export function phaseInputPerturbationNorms(
    h0: Float32Array,
    N: number,
    L: number,
    phaseErrorL2: number,
): { x: number; z: number } {
    const dk = (2 * Math.PI) / L;
    let xOperator = 0;
    let zOperator = 0;
    for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
            const index = y * N + x;
            const neg = ((N - y) % N) * N + ((N - x) % N);
            const modeNorm = Math.hypot(
                h0[index * 2],
                h0[index * 2 + 1],
                h0[neg * 2],
                h0[neg * 2 + 1],
            );
            const kx = Math.abs((x <= N / 2 ? x : x - N) * dk);
            const kz = Math.abs((y <= N / 2 ? y : y - N) * dk);
            xOperator = Math.max(xOperator, kx * modeNorm);
            zOperator = Math.max(zOperator, kz * modeNorm);
        }
    }
    return { x: N * xOperator * phaseErrorL2, z: N * zOperator * phaseErrorL2 };
}
