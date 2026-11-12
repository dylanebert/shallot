import { expect, test } from "bun:test";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { openBasisImage } from "./basisu";

const root = resolve(import.meta.dir, "../../..");

test("JavaScript entry refuses malformed input, invalid levels and use after disposal", async () => {
    expect(() => openBasisImage(new Uint8Array(80))).toThrow("identifier");
    const image = openBasisImage(
        await readFile(resolve(root, "crates/basisu/tests/fixtures/uastc-none.ktx2")),
    );
    expect(() => image.transcode(-1, "astc")).toThrow("level");
    expect(() => image.transcode(0, "etc1")).toThrow("target");
    const first = image.transcode(0, "rgba8");
    const saved = first.slice();
    image.transcode(0, "astc");
    image.dispose();
    image.dispose();
    expect(first).toEqual(saved);
    expect(() => image.transcode(0, "astc")).toThrow("disposed");
});
