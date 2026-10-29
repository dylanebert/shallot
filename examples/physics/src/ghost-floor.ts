const f = Math.fround;
type Point = [number, number, number];
const SRC = f(0.0254);
/** The two s&box collision chunks, in meters, with their cooked-map T-junctions. */
export function ghostFloor(chunk: number) {
    const vertices: number[] = [],
        indices: number[] = [];
    const x0U = chunk === 0 ? -256 : 0,
        x1U = chunk === 0 ? 0 : 256;
    function triangle(a: Point, b: Point, c: Point) {
        const base = vertices.length / 3;
        vertices.push(...a, ...b, ...c);
        indices.push(base, base + 1, base + 2);
    }
    function point(x: number, y: number, z: number): Point {
        return [f(SRC * x), f(SRC * y), f(SRC * z)];
    }
    function clip(a: number, b: number): [number, number] | null {
        const low = Math.max(a, x0U),
            high = Math.min(b, x1U);
        return f(high - low) > f(0.01) ? [low, high] : null;
    }
    function patch(x0: number, x1: number, z0: number, z1: number, y: number, cell: number) {
        if (f(x1 - x0) < f(0.01)) return;
        const nx = Math.trunc(f(f(f(x1 - x0) / cell) + f(0.99))),
            nz = Math.trunc(f(f(f(z1 - z0) / cell) + f(0.99)));
        for (let ix = 0; ix < nx; ix++)
            for (let iz = 0; iz < nz; iz++) {
                const cx0 = f(x0 + f(f(f(x1 - x0) * ix) / nx)),
                    cx1 = f(x0 + f(f(f(x1 - x0) * (ix + 1)) / nx));
                const cz0 = f(z0 + f(f(f(z1 - z0) * iz) / nz)),
                    cz1 = f(z0 + f(f(f(z1 - z0) * (iz + 1)) / nz));
                const a = point(cx0, y, cz0),
                    b = point(cx1, y, cz0),
                    c = point(cx1, y, cz1),
                    d = point(cx0, y, cz1);
                if ((ix + iz) & 1) {
                    triangle(a, d, c);
                    triangle(a, c, b);
                } else {
                    triangle(a, d, b);
                    triangle(b, d, c);
                }
            }
    }
    function strip(
        xLow: number,
        yLow: number,
        xHigh: number,
        yHigh: number,
        cell: number,
        facing = 0,
    ) {
        const nz = Math.trunc(f(f(128 / cell) + f(0.99)));
        for (let iz = 0; iz < nz; iz++) {
            const z0 = f(-64 + f(f(128 * iz) / nz)),
                z1 = f(-64 + f(f(128 * (iz + 1)) / nz));
            const l0 = point(xLow, yLow, z0),
                l1 = point(xLow, yLow, z1),
                h0 = point(xHigh, yHigh, z0),
                h1 = point(xHigh, yHigh, z1);
            if (facing < 0) {
                triangle(l0, h0, h1);
                triangle(l0, h1, l1);
            } else {
                triangle(l0, l1, h1);
                triangle(l0, h1, h0);
            }
        }
    }
    function hash(x: number) {
        x ^= x >>> 16;
        x = Math.imul(x, 0x7feb352d);
        x ^= x >>> 15;
        x = Math.imul(x, 0x846ca68b);
        x ^= x >>> 16;
        return x >>> 0;
    }
    for (const [a, b] of [
        [-256, -94],
        [94, 256],
    ]) {
        const span = clip(a!, b!);
        if (!span) continue;
        for (let tx = span[0]; tx < span[1]; tx = f(tx + 32))
            for (let tz = -64; tz < 64; tz += 32) {
                const h = hash(
                    Math.imul(Math.trunc(tx), 73856093) ^
                        Math.imul(tz, 19349663) ^
                        Math.imul(chunk, 2654435761),
                );
                patch(tx, Math.min(f(tx + 32), span[1]), tz, tz + 32, 0, [4, 8, 16][h % 3]!);
            }
    }
    for (let k = 0; k < 9; k++) {
        const bx = f(-94 + f(22 * k)),
            left = k > 0,
            right = k < 8;
        const top = clip(left ? f(bx + 1.5) : bx, right ? f(f(bx + 12) - 1.5) : f(bx + 12));
        if (top) patch(...top, -64, 64, 0, 8);
        if (left && bx >= x0U && bx < x1U) strip(bx, -1, f(bx + 1.5), 0, 8);
        if (right && f(bx + 12) > x0U && f(bx + 12) <= x1U)
            strip(f(bx + 12), -1, f(f(bx + 12) - 1.5), 0, 8);
        if (right) {
            const l = f(bx + 12),
                r = f(bx + 22);
            if (l >= x0U && l < x1U) strip(l, -24, l, -1, 16, 1);
            if (r > x0U && r <= x1U) strip(r, -24, r, -1, 16, -1);
            const pit = clip(l, r);
            if (pit) patch(...pit, -64, 64, -24, 16);
        }
    }
    return {
        vertices: new Float32Array(vertices),
        indices: new Uint32Array(indices),
        weldVertices: true,
        weldTolerance: 0.005,
        identifyEdges: true,
    };
}
