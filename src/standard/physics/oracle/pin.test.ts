import { expect, test } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { BOX3D_PIN } from "./pin";

test("every corpus lane was generated at the pinned Box3D revision and tree", () => {
    const corpus = join(import.meta.dir, "box3d");
    const lanes = readdirSync(corpus).sort();
    expect(lanes).toEqual(["v1", "v2", "v3", "v6", "v7", "v8"]);
    for (const lane of lanes) {
        const { upstream } = JSON.parse(
            readFileSync(join(corpus, lane, "manifest.json"), "utf8"),
        ) as {
            upstream: { sha: string; tree: string };
        };
        expect({ lane, sha: upstream.sha, tree: upstream.tree }).toEqual({
            lane,
            sha: BOX3D_PIN.revision,
            tree: BOX3D_PIN.tree,
        });
    }
});
