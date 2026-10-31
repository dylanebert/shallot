import { afterAll, beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

setDefaultTimeout(TIMEOUT.node);

let cache = "";
let results: { exit: number; stdout: string; stderr: string }[] = [];

beforeAll(async () => {
    cache = mkdtempSync(join(tmpdir(), "box3d-native-cache-"));
    const source = `
        import { nativeBinary, run } from ${JSON.stringify(join(import.meta.dir, "native.ts"))};
        import { join } from "node:path";
        const binary = nativeBinary("world-mover.c", undefined, join(process.env.BOX3D, "shared/benchmarks.c"));
        process.stdout.write(run([binary]));
    `;
    const processes = Array.from({ length: 2 }, () => Bun.spawn([process.execPath, "-e", source], {
        env: { ...process.env, TMPDIR: cache },
        stdout: "pipe",
        stderr: "pipe",
    }));
    results = await Promise.all(processes.map(async process => {
        const [exit, stdout, stderr] = await Promise.all([
            process.exited,
            new Response(process.stdout).text(),
            new Response(process.stderr).text(),
        ]);
        return { exit, stdout, stderr };
    }));
}, TIMEOUT.startup);

afterAll(() => {
    if (cache) rmSync(cache, { recursive: true, force: true });
});

test("concurrent cold native builds publish complete libraries and executables", () => {
    for (const result of results) expect(result.stderr).toBe("");
    for (const result of results) expect(result.exit).toBe(0);
    expect(results[0].stdout.length).toBeGreaterThan(0);
    expect(results[1].stdout).toBe(results[0].stdout);
});
