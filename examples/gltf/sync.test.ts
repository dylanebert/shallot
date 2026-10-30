import { expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { syncFiles } from "./sync";

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
