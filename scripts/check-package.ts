import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { gunzipSync } from "node:zlib";

const MAX_UNPACKED_BYTES = 8_000_000;
const root = resolve(import.meta.dir, "..");
const manifest = JSON.parse(readFileSync(resolve(root, "package.json"), "utf8"));
const packageFile = `${manifest.name.replace(/^@/, "").replace("/", "-")}-${manifest.version}.tgz`;
const scratch = mkdtempSync(join(tmpdir(), "shallot-package-lint-"));
const tarball = join(scratch, packageFile);

function unpackedSize(tarball: string): number {
    const archive = gunzipSync(readFileSync(tarball));
    let bytes = 0;
    for (let offset = 0; offset + 512 <= archive.length; ) {
        const header = archive.subarray(offset, offset + 512);
        if (header.every((byte) => byte === 0)) break;
        const field = header.subarray(124, 136).toString("ascii").replace(/\0.*$/, "").trim();
        const size = field === "" ? 0 : Number.parseInt(field, 8);
        if (!Number.isSafeInteger(size) || size < 0)
            throw new Error(`invalid tar entry size: ${JSON.stringify(field)}`);
        if (header[156] === 0 || header[156] === 48) bytes += size;
        offset += 512 + Math.ceil(size / 512) * 512;
    }
    return bytes;
}

function run(label: string, command: string[]): string {
    const start = performance.now();
    const result = Bun.spawnSync(command, {
        cwd: root,
        stdout: "pipe",
        stderr: "pipe",
    });
    const seconds = ((performance.now() - start) / 1000).toFixed(1);
    if (!result.success) {
        process.stdout.write(result.stdout);
        process.stderr.write(result.stderr);
        throw new Error(`${label} failed (${seconds}s)`);
    }
    return seconds;
}

try {
    const packSeconds = run("bun pm pack", [
        "bun",
        "pm",
        "pack",
        "--destination",
        scratch,
        "--quiet",
    ]);
    if (!existsSync(tarball)) throw new Error(`packed tarball is missing: ${tarball}`);

    const size = unpackedSize(tarball);
    if (size > MAX_UNPACKED_BYTES)
        throw new Error(
            `packed tarball unpacked size ${size.toLocaleString()} bytes exceeds ceiling ${MAX_UNPACKED_BYTES.toLocaleString()} bytes`,
        );
    console.log(
        `✓ packed unpacked size ${size.toLocaleString()} bytes (ceiling ${MAX_UNPACKED_BYTES.toLocaleString()} bytes)`,
    );

    const publintSeconds = run("publint", [
        "publint",
        "run",
        tarball,
        "--strict",
        "--pack",
        "false",
    ]);
    // ATTW's ignore is global: it accepts the TypeScript-source Node16 gap owned by publish.md,
    // but could hide a broken bundler import too. The packed-install test typechecks every export
    // under bundler resolution to cover that loss.
    const attwSeconds = run("@arethetypeswrong/cli", ["attw", tarball]);

    console.log(
        `✓ one packed tarball shared by publint and @arethetypeswrong/cli (${packSeconds}s)`,
    );
    console.log(`✓ publint (${publintSeconds}s)`);
    console.log(`✓ @arethetypeswrong/cli (${attwSeconds}s)`);
} finally {
    rmSync(scratch, { recursive: true, force: true });
}
