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

check(
    "immutable Box3D public oracle bundle is selected and intact",
    {
        claim: "Box3D public oracle bundle records target separately from conformance and all generated evidence digests match",
        size: "integration",
    },
    () => {
        if (current.target !== manifest.bundle.upstreamSha)
            throw new Error("current target does not match bundle upstream SHA");
        if (current.conformsThrough === current.target)
            throw new Error("publishing target advanced conformance");
        if (manifest.bundle.schema !== "v1" && manifest.bundle.schema !== "v2" && manifest.bundle.schema !== "v3" && manifest.bundle.schema !== "v4") throw new Error("unsupported oracle schema");
        for (const [name, expected] of Object.entries(manifest.fileDigests)) {
            if (digest(join(bundle, name)) !== expected)
                throw new Error(`bundle digest mismatch: ${name}`);
        }
        if (manifest.bundle.schema === "v4") {
            const membership = JSON.parse(readFileSync(join(bundle, "membership.json"), "utf8")) as { count?: number; names?: string[]; exact?: boolean };
            if (membership.count !== 53 || membership.exact !== true || membership.names?.length !== 53) throw new Error("scenario membership is not exact 53-name coverage");
        }
    },
);
