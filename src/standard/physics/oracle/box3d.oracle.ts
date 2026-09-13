import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { check } from "../../../harness/check";

const root = join(import.meta.dir, "box3d");
const current = JSON.parse(readFileSync(join(root, "current.json"), "utf8")) as {
    target: string;
    conformsThrough: string;
    bundle: string;
};
const bundle = join(root, current.bundle);
const manifest = JSON.parse(readFileSync(join(bundle, "manifest.json"), "utf8")) as {
    bundle: { upstreamSha: string; schema: string };
    fileDigests: Record<string, string>;
};

function digest(path: string): string {
    return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function assertBundle(): void {
    if (current.target !== manifest.bundle.upstreamSha)
        throw new Error("current target does not match bundle upstream SHA");
    if (current.conformsThrough === current.target)
        throw new Error("publishing target advanced conformance");
    if (
        manifest.bundle.schema !== "v1" &&
        manifest.bundle.schema !== "v2" &&
        manifest.bundle.schema !== "v3"
    )
        throw new Error("unsupported oracle schema");
    for (const [name, expected] of Object.entries(manifest.fileDigests)) {
        if (digest(join(bundle, name)) !== expected)
            throw new Error(`bundle digest mismatch: ${name}`);
    }
}

check(
    "immutable Box3D public oracle bundle is selected and intact",
    {
        claim: "Box3D public oracle bundle records target separately from conformance and all generated evidence digests match",
        size: "integration",
    },
    assertBundle,
);

check(
    "Box3D known differences remain explicit",
    { claim: "box3d-known-differences", size: "integration" },
    () => {
        assertBundle();
        if (current.conformsThrough === current.target)
            throw new Error("known differences disappeared without advancing the oracle target");
    },
);
