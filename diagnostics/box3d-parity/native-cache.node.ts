import { expect, setDefaultTimeout, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

setDefaultTimeout(180_000);

test("concurrent cold native builds publish complete libraries and executables", async () => {
    const cache = mkdtempSync(join(tmpdir(), "box3d-native-cache-"));
    const source = `
        import { nativeBinary, run } from ${JSON.stringify(join(import.meta.dir, "native.ts"))};
        import { join } from "node:path";
        const binary = nativeBinary("world-mover.c", undefined, join(process.env.BOX3D, "shared/benchmarks.c"));
        process.stdout.write(run([binary]));
    `;
    try {
        const processes = Array.from({ length: 2 }, () => Bun.spawn([process.execPath, "-e", source], {
            env: { ...process.env, TMPDIR: cache },
            stdout: "pipe",
            stderr: "pipe",
        }));
        const results = await Promise.all(processes.map(async process => {
            const [exit, stdout, stderr] = await Promise.all([
                process.exited,
                new Response(process.stdout).text(),
                new Response(process.stderr).text(),
            ]);
            return { exit, stdout, stderr };
        }));
        for (const result of results) expect(result.stderr).toBe("");
        for (const result of results) expect(result.exit).toBe(0);
        expect(results[0].stdout.length).toBeGreaterThan(0);
        expect(results[1].stdout).toBe(results[0].stdout);
    } finally {
        rmSync(cache, { recursive: true, force: true });
    }
});
