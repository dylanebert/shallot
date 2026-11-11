import { expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import inputs from "../../../crates/basisu/tests/records/inputs.json";
import { syncFiles } from "../../../examples/gltf/sync";
import { type BasisTarget, openBasisImage } from "./basisu";

const root = resolve(import.meta.dir, "../../..");
const corpus = resolve(root, "test-results/basisu-assets");

test("JavaScript entry copies every recorded ETC1S and UASTC target bit-equal to the pinned reference", async () => {
    await syncFiles(
        inputs.files,
        new URL(
            `https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/${inputs.commit}/Models/`,
        ),
        corpus,
    );
    const records = (
        await Promise.all(
            ["etc1s", "uastc"].map((name) =>
                readFile(resolve(root, `crates/basisu/tests/records/${name}.tsv`), "utf8"),
            ),
        )
    ).flatMap((records) => records.trim().split("\n").slice(1));
    let image: ReturnType<typeof openBasisImage> | undefined;
    let current = "";
    let count = 0;
    try {
        for (const line of records) {
            const [asset, level, target, length, hash] = line.split("\t");
            if (asset !== current) {
                image?.dispose();
                const [model, file] = asset.split("/");
                const path =
                    model === "fixtures"
                        ? resolve(root, `crates/basisu/tests/fixtures/${file}.ktx2`)
                        : resolve(corpus, `${model}/glTF-KTX-BasisU/${file}.ktx2`);
                image = openBasisImage(await readFile(path));
                current = asset;
            }
            const output = image!.transcode(Number(level), target as BasisTarget);
            expect(
                [output.length, createHash("sha256").update(output).digest("hex")],
                `${asset} level ${level} ${target}`,
            ).toEqual([Number(length), hash]);
            count++;
        }
    } finally {
        image?.dispose();
    }
    expect(count).toBe(702);
});

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
