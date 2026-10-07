import { expect, setDefaultTimeout, test } from "bun:test";
import { nativeSseOutput } from "./native-evidence";
import { oracleWasm } from "./oracle-kernel";

setDefaultTimeout(180_000);

const wasm = await WebAssembly.compile(oracleWasm);
const imports: Record<string, Record<string, () => number>> = {};
for (const entry of WebAssembly.Module.imports(wasm)) {
    if (entry.kind !== "function") throw new Error(`unexpected oracle import ${entry.name}`);
    (imports[entry.module] ??= {})[entry.name] = () => {
        throw new Error(`unexpected oracle call ${entry.module}.${entry.name}`);
    };
}
const instance = await WebAssembly.instantiate(wasm, imports);
const k = instance.exports as unknown as {
    memory: WebAssembly.Memory;
    box3dOracleInput(): number;
    box3dOracleOutput(): number;
    box3dOracleRun(operation: number): number;
};
const hex = (v: number) => (v >>> 0).toString(16).padStart(8, "0");
const bits = (v: number) => new Uint32Array(new Float32Array([v]).buffer)[0];
let seed = 0x47d7f7cc;
function next(): number {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return seed >>> 0;
}
type Case = { operation: number; words: number[]; label?: string };
const cases: Case[] = [];
for (let exponent = 0; exponent < 255; ++exponent) {
    for (const mantissa of [0, 1, 0x3fffff, 0x7ffffe, 0x7fffff, next() & 0x7fffff]) {
        for (const sign of [0, 0x80000000]) {
            const word = (sign | (exponent << 23) | mantissa) >>> 0;
            cases.push({ operation: 1, words: [word] });
        }
    }
}
for (const magnitude of [0, 1e-40, 1e-30, 1e-20, 1e-19, 1e-18, 1, 1e10, 1e20]) {
    cases.push({ operation: 0, words: [bits(magnitude), 0, 0] });
    cases.push({ operation: 0, words: [bits(magnitude), bits(-magnitude), bits(magnitude)] });
}
for (let i = 0; i < 256; ++i) {
    const f = () => bits(((next() / 0x100000000) * 2 - 1) * 100);
    cases.push({ operation: 0, words: [f(), f(), f()] });
    cases.push({ operation: 2, words: Array.from({ length: 8 }, f) });
}
cases.push({ operation: 2, words: [0, 0x80000000, 0, 0x80000000, 0x80000000, 0, 0, 0x80000000] });

const proxies = [
    [[0, 0, 0]],
    [
        [0, -1, 0],
        [0, 1, 0],
    ],
    [
        [-1, 0, -1],
        [1, 0, -1],
        [0, 0, 1],
    ],
    [
        [-1, -1, -1],
        [1, -1, 1],
        [-1, 1, 1],
        [1, 1, -1],
    ],
];
for (const a of proxies)
    for (const b of proxies) {
        for (const offset of [0, 1e-20, 0.01, 1, 3, 20]) {
            for (const operation of [3, 4, 5]) {
                const words = Array<number>(834).fill(0);
                words[0] = a.length;
                words[1] = bits(0.25);
                words[2] = b.length;
                words[3] = bits(0.25);
                words[4] = bits(offset);
                words[10] = bits(1);
                words[11] = 1;
                words[12] = bits(-20);
                words[15] = bits(1);
                words.splice(32, a.length * 3, ...a.flat().map(bits));
                words.splice(416, b.length * 3, ...b.flat().map(bits));
                words[812] = words[829] = bits(1);
                words[816] = words[833] = bits(1);
                words[820] = bits(offset);
                words[823] = bits(offset - 20);
                cases.push({
                    operation,
                    words,
                    label: `proxy-${operation}-${a.length}-${b.length}-${offset}`,
                });
            }
        }
    }

for (const a of proxies)
    for (const b of proxies) {
        const words = [
            ...cases.find((c) => c.label === `proxy-5-${a.length}-${b.length}-3`)!.words,
        ];
        words[815] = bits(Math.sin(0.23));
        words[816] = bits(Math.cos(0.23));
        words[830] = bits(Math.sin(0.47));
        words[833] = bits(Math.cos(0.47));
        cases.push({ operation: 5, words, label: `rotating-toi-${a.length}-${b.length}` });
    }
{
    const words = [...cases.find((c) => c.label === "proxy-5-1-1-3")!.words];
    words[823] = bits(23);
    cases.push({ operation: 5, words, label: "separated-toi-moving-away" });
}
for (const scale of [1e8, 1e10, 1e12]) {
    const words = Array<number>(834).fill(0);
    words[0] = 2;
    words[2] = 1;
    words[10] = bits(1);
    words.splice(32, 6, ...[-scale, 0, 0, scale, 0, 0].map(bits));
    words.splice(416, 3, ...[0, scale, 0].map(bits));
    cases.push({ operation: 3, words, label: `invalid-normal-${scale}` });
}

const hullWords = nativeSseOutput("functions.c", "99 0\n", ["hull image provisioning"])
    .trim()
    .split(" ")
    .slice(1)
    .map((h) => Number.parseInt(h, 16));
const sizeA = hullWords[0];
const imageA = hullWords.slice(1, 1 + sizeA);
const sizeB = hullWords[1 + sizeA];
const imageB = hullWords.slice(2 + sizeA, 2 + sizeA + sizeB);
for (let operation = 10; operation <= 18; ++operation) {
    for (let i = 0; i < 512; ++i) {
        const words = Array<number>(2000).fill(0);
        const put = (offset: number, values: number[]) =>
            words.splice(offset, values.length, ...values.map(bits));
        const random = () => (next() / 0x100000000) * 4 - 2;
        words[0] = 32;
        const rotation = [random(), random(), random(), random()];
        const length = Math.hypot(...rotation);
        put(4, [random(), random(), random(), ...rotation.map((v) => v / length)]);
        if (operation < 16) {
            words[2] = 8;
            words[3] = bits(13.5);
        }
        put(12, [0, 0.3, 0, 0.5]);
        put(16, [0, 0, 0, 0.5]);
        put(20, [-0.7, 0.3, 0, 0.7, 0.3, 0, 0.5]);
        put(27, [random(), random(), random(), random(), random(), random(), 0.5]);
        put(34, [-2, 0, -2, 0, 0, 2, 2, 0, -2]);
        if (i < 16) put(4, [i / 100, 1.5 + i / 100, 0, 0, 0, 0, 1]);
        if (operation >= 16) {
            put(12, [random(), Math.abs(random()), random(), 0.5]);
            put(20, [
                random(),
                Math.abs(random()),
                random(),
                random(),
                Math.abs(random()),
                random(),
                0.5,
            ]);
        }
        if (operation === 18 && i >= 16) {
            const value = (word: number) => new Float32Array(new Uint32Array([word]).buffer)[0];
            const q = words.slice(7, 11).map(value),
                p = words.slice(4, 7).map(value);
            const cross = (a: number[], b: number[]) => [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ];
            const transformed = [
                [-2, 0, -2],
                [0, 0, 2],
                [2, 0, -2],
            ].flatMap((v) => {
                const t = cross(q, v).map((n) => 2 * n);
                const u = cross(q, t);
                return v.map((n, j) => n + q[3] * t[j] + u[j] + p[j]);
            });
            put(34, transformed);
        }
        if (operation === 14 && i < 16) {
            put(4, [0, 0, 0, 0, 0, 0, 1]);
            put(27, [1, 1 + i / 10000, 0, 2, 1 + i / 10000, 0, 0.5]);
        }
        words.splice(1000, sizeA, ...imageA);
        words.splice(1500, sizeB, ...imageB);
        cases.push({ operation, words, label: `manifold-${operation}-${i}` });
    }
}

{
    const words = [...cases.find((c) => c.label === "manifold-10-0")!.words];
    words[1] = 1;
    words[4] = bits(100);
    cases.push({ operation: 10, words, label: "sphere-sphere-retained-early-exit" });
}
for (let along = 0; along < 3; ++along) {
    for (let side = 0; side < 3; ++side) {
        if (along === side) continue;
        for (const direction of [-1, 1])
            for (const boundary of [-1, 1]) {
                const template = cases.find((c) => c.label === "manifold-14-0")!;
                const words = [...template.words];
                const p = [0, 0, 0],
                    q = [0, 0, 0];
                p[along] = direction;
                q[along] = 2 * direction;
                p[side] = q[side] = boundary;
                words.splice(27, 7, ...[...p, ...q, 0.5].map(bits));
                cases.push({
                    operation: 14,
                    words,
                    label: `null-axis-${along}-${side}-${direction}-${boundary}`,
                });
            }
    }
}

for (const mode of [6, 7, 8]) {
    for (const original of cases.filter((c) => c.operation === 15).slice(0, 64)) {
        const words = [...original.words];
        words[45] = mode;
        cases.push({ operation: 15, words, label: `manual-axis-${mode}-${original.label}` });
    }
}

for (const [label, x] of [
    ["ties", [1, 1, 1, 1, 1, 1, 1, 1]],
    ["signed-zero", [0, -0, 0, -0, -0, 0, -0, 0]],
    ["first-lane", [2, 1, 1, 1, 1, 1, 1, 1]],
    ["last-lane", [1, 1, 1, 1, 1, 1, 1, 2]],
] as const) {
    cases.push({ operation: 6, label: `support-${label}`,
        words: [1, 0, 0, 4, ...x, ...Array(16).fill(0)].map(bits) });
}

const native = nativeSseOutput(
    "functions.c",
    cases.map((c) => `${c.operation} ${c.words.length} ${c.words.map(hex).join(" ")}`).join("\n"),
    cases.map((c, i) => `${c.operation}:${c.label ?? i}`),
)
    .trim()
    .split("\n")
    .map((line) => line.split(" ").slice(1));
if (native.length !== cases.length) throw new Error("native returned the wrong case count");
const warm: Case[] = [];
for (let i = 0; i < cases.length && warm.length < 64; ++i) {
    if (cases[i].operation !== 15 || native[i].at(-4) !== "00000004") continue;
    const words = [...cases[i].words];
    words.splice(44, 5, ...native[i].slice(-5).map((h) => Number.parseInt(h, 16)));
    const rotation = Array.from({ length: 4 }, () => (next() / 0x100000000) * 2 - 1);
    const length = Math.hypot(...rotation);
    words.splice(4, 7, ...[10, 0, 0, ...rotation.map((v) => v / length)].map(bits));
    warm.push({ operation: 15, words, label: `cached-edge-${cases[i].label}` });
}
const warmRows = nativeSseOutput(
    "functions.c",
    warm.map((c) => `${c.operation} ${c.words.length} ${c.words.map(hex).join(" ")}`).join("\n"),
    warm.map((c) => c.label!),
);
cases.push(...warm);
native.push(
    ...warmRows
        .trim()
        .split("\n")
        .map((line) => line.split(" ").slice(1)),
);
const results = cases.map((c, i) => {
    new Uint32Array(k.memory.buffer, k.box3dOracleInput(), c.words.length).set(c.words);
    let actual: string[];
    try {
        const count = k.box3dOracleRun(c.operation);
        actual = [...new Uint32Array(k.memory.buffer, k.box3dOracleOutput(), count)].map(hex);
    } catch (error) {
        actual = [String(error)];
    }
    const expected = [...native[i]];
    const actualMetadata =
        c.operation >= 10 ? actual.splice(6 + Number.parseInt(actual[0], 16) * 6, 1) : [];
    const nativeMetadata =
        c.operation >= 10 ? expected.splice(6 + Number.parseInt(expected[0], 16) * 6, 9) : [];
    return { ...c, actual, native: expected, actualMetadata, nativeMetadata };
});
test.todo("manifold.rs:79-96; triangle_manifold.rs:348: local manifold returns triangle normal, index, vertices and flags", () => {
    const rows = results.filter((r) => r.operation >= 10);
    const mismatches = rows.filter(
        (r) => JSON.stringify(r.actualMetadata) !== JSON.stringify(r.nativeMetadata),
    );
    expect(mismatches.map((r) => r.label)).toEqual([]);
});
for (const [operation, entry] of [
    [0, "math.rs:908-915 vs math_functions.h:290-302 normalize"],
    [1, "math.rs:75-113 vs math_functions.h:213-216 unwind remainder"],
    [2, "wide.rs:13-17 vs simd.h:633-637 sym_clamp operand order"],
    [
        3,
        "distance.rs:745-750,777-783 early iterations; distance.rs:824-838 witness/normal/cache order",
    ],
    [16, "triangle_manifold.rs:578-579 reciprocal barycentrics; Box3D math_functions.c:554-555"],
] as const) {
    const rows = results.filter((r) => r.operation === operation);
    const check = operation <= 2 ? test : test.todo;
    check(`${entry}: seeded boundary outputs equal Box3D SSE2 bits`, () => {
        compare(rows);
    });
}
for (const [operation, name] of [
    [4, "shape cast"],
    [5, "TOI"],
    [11, "capsule-sphere"],
    [12, "hull-sphere"],
    [13, "capsule-capsule"],
    [17, "triangle-capsule"],
    [18, "triangle-hull"],
] as const) {
    test(`${name}: generated point-cloud boundary outputs equal Box3D SSE2 bits`, () => {
        const rows = results.filter((r) => r.operation === operation);
        for (const row of rows) expect(row.actual, row.label).toEqual(row.native);
    });
}
function compare(rows: typeof results) {
    expect(rows.length).toBeGreaterThan(0);
    for (const row of rows)
        expect(
            row.actual,
            `${row.label ?? row.operation}: ${row.words.slice(0, 64).map(hex).join(" ")}`,
        ).toEqual(row.native);
}
test("hull support lane ties, signed zero and first/last winners equal Box3D SSE2 bits", () => {
    compare(results.filter((r) => r.operation === 6));
});
test("normalize ordinary finite vectors equal Box3D SSE2 bits", () => {
    compare(
        results.filter(
            (r) =>
                r.operation === 0 &&
                r.words.some((w) => (w & 0x7fffffff) >= bits(1)) &&
                r.words.every((w) => (w & 0x7fffffff) <= bits(1e10)),
        ),
    );
});
test("unwind ordinary finite angles equal Box3D SSE2 bits", () => {
    compare(
        results.filter(
            (r) =>
                r.operation === 1 &&
                (r.words[0] & 0x7fffffff) >= bits(0.125) &&
                (r.words[0] & 0x7fffffff) <= bits(3),
        ),
    );
});
test("symmetric clamp unequal nonzero finite operands equal Box3D SSE2 bits", () => {
    compare(
        results.filter((r) => r.operation === 2 && r.words.every((w) => (w & 0x7fffffff) !== 0)),
    );
});
test("distance separated point clouds equal Box3D SSE2 bits", () => {
    compare(
        results.filter(
            (r) => r.operation === 3 && r.label?.startsWith("proxy-") && r.words[4] === bits(20),
        ),
    );
});
test("sphere-sphere generated fresh manifolds equal Box3D SSE2 bits", () => {
    compare(
        results.filter(
            (r) => r.operation === 10 && r.label !== "sphere-sphere-retained-early-exit",
        ),
    );
});
test.todo("distance.rs:745-750 vs distance.c:847-851: contained tetrahedron writes iterations", () => {
    compare(results.filter((r) => r.label === "proxy-3-4-1-0"));
});
test.todo("distance.rs:777-783 vs distance.c:947-951: line containment writes iterations", () => {
    compare(results.filter((r) => r.label === "proxy-3-2-1-0"));
});
test.todo("distance.rs:824-838 vs distance.c:997-1011: invalid normal retains witness and iteration outputs", () => {
    compare(results.filter((r) => r.label?.startsWith("invalid-normal-")));
});
test.todo("manifold.rs:794 vs convex_manifold.c:384-389: sphere-sphere separation retains caller manifold", () => {
    compare(results.filter((r) => r.label === "sphere-sphere-retained-early-exit"));
});
test("manifold.rs:383-388 vs convex_manifold.c:36: endpoint on clip plane is not another intersection", () => {
    compare(
        results.filter(
            (r) => r.operation === 14 && /^manifold-14-(?:[1-9]|1[0-5])$/.test(r.label ?? ""),
        ),
    );
});
test.todo("manifold.rs:383-388,558-559,1424-1431 vs convex_manifold.c:36,127-128,931-934: clipped-away face with null edge", () => {
    compare(results.filter((r) => r.label?.startsWith("null-axis-")));
});
test.todo("manifold.rs:1424-1431 vs convex_manifold.c:939: hull-capsule face/edge tolerance chooses the native point count", () => {
    const row = results.find((r) => r.label === "manifold-14-29")!;
    expect(row.actual[0]).toBe(row.native[0]);
});
test.todo("manifold.rs:1252-1261 vs convex_manifold.c:782: hull-capsule edge uses query normal", () => {
    compare(results.filter((r) => ["manifold-14-62", "manifold-14-100"].includes(r.label ?? "")));
});
test.todo("manifold.rs:1424-1431,1252-1261: generated hull-capsule edge decisions and normals equal Box3D SSE2 bits", () => {
    compare(
        results.filter(
            (r) =>
                r.operation === 14 &&
                r.label?.startsWith("manifold-") &&
                Number(r.label.split("-")[2]) >= 16,
        ),
    );
});
test.todo("manifold.rs:2240-2257 vs convex_manifold.c:2089: hull-hull fallback copies complete metadata", () => {
    compare(results.filter((r) => r.operation === 15 && r.label?.startsWith("manifold-")));
});
test.todo("manifold.rs:2118-2157 vs convex_manifold.c:1960-1986: manual axes use b3ComputeSeparatingAxis", () => {
    compare(results.filter((r) => r.label?.startsWith("manual-axis-")));
});

test("manifold.rs:2072-2080 vs convex_manifold.c:1916: generated cached hull-hull edge boundaries stay equal", () => {
    compare(results.filter((r) => r.label?.startsWith("cached-edge-")));
});
