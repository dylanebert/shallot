import { expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import manifest from "./assets.json";
import { devModels, pickerModels, pickerPaths } from "./dev-assets";
import { syncFiles } from "./sync";

test("dev server syncs exactly the picker's whole manifest models", () => {
    const expected = manifest.models.filter((model) =>
        Object.values(pickerPaths).some((path) =>
            model.files.some((file) => `assets/${file.path}` === path),
        ),
    );
    expect(devModels.map((model) => model.name).sort()).toEqual(
        expected.map((model) => model.name).sort(),
    );
    expect(devModels).toHaveLength(25);
    expect(devModels.flatMap((model) => model.files)).toHaveLength(228);
    for (const model of devModels) {
        expect(model.files.some((file) => file.path === model.license)).toBe(true);
    }
    expect(pickerModels({ written: "models/SkinnedNormal.gltf" })).toEqual([]);
    expect(() => pickerModels({ NewScene: "assets/missing.gltf" })).toThrow(
        "NewScene: no manifest model or written fixture for assets/missing.gltf",
    );
});

test("sync bounds concurrent downloads and writes repeated manifest paths once", async () => {
    const root = await mkdtemp(join(tmpdir(), "gltf-concurrent-"));
    const bytes = "model bytes";
    const sha256 = createHash("sha256").update(bytes).digest("hex");
    const files = Array.from({ length: 20 }, (_, i) => ({ path: `${i}.bin`, sha256 }));
    let active = 0;
    let peak = 0;
    let requests = 0;
    const server = Bun.serve({
        port: 0,
        async fetch() {
            requests++;
            peak = Math.max(peak, ++active);
            await Bun.sleep(10);
            active--;
            return new Response(bytes);
        },
    });
    try {
        await syncFiles([...files, ...files], server.url, root);
        expect(peak).toBeGreaterThan(1);
        expect(peak).toBeLessThanOrEqual(8);
        expect(requests).toBe(files.length);
        expect((await readdir(root)).sort()).toEqual(files.map((file) => file.path).sort());
        for (const file of files) expect(await readFile(join(root, file.path), "utf8")).toBe(bytes);
    } finally {
        server.stop(true);
        await rm(root, { recursive: true, force: true });
    }
});

test("sync fetches missing files, reuses only verified files, and names both hashes on refusal", async () => {
    const root = await mkdtemp(join(tmpdir(), "gltf-sync-"));
    const bytes = "model bytes";
    const sha256 = createHash("sha256").update(bytes).digest("hex");
    const wrong = createHash("sha256").update("wrong bytes").digest("hex");
    const file = { path: "model.bin", sha256 };
    const destination = join(root, "cache");
    const cached = join(destination, file.path);
    const source = pathToFileURL(`${root}/`);
    try {
        await writeFile(join(root, file.path), bytes);
        await syncFiles([file], source, destination);
        expect(await readFile(cached, "utf8")).toBe(bytes);
        await rm(join(root, file.path));
        await syncFiles([file], source, destination);
        expect(await readFile(cached, "utf8")).toBe(bytes);

        const message = `${file.path}: SHA-256 mismatch: expected ${sha256}, received ${wrong}`;
        await writeFile(cached, "wrong bytes");
        await expect(syncFiles([file], source, destination)).rejects.toThrow(message);
        expect(await readFile(cached, "utf8")).toBe("wrong bytes");

        await rm(cached);
        await writeFile(join(root, file.path), "wrong bytes");
        await expect(syncFiles([file], source, destination)).rejects.toThrow(message);
        expect(await Bun.file(cached).exists()).toBe(false);
    } finally {
        await rm(root, { recursive: true, force: true });
    }
});
