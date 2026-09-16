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
};
check(
    "the selected Box3D oracle bundle is the fully conformed target",
    {
        claim: "Box3D public oracle bundle records the fully conformed target",
        size: "integration",
    },
    () => {
        if (current.target !== manifest.bundle.upstreamSha)
            throw new Error("current target does not match bundle upstream SHA");
        if (current.conformsThrough !== current.target)
            throw new Error("selected immutable target is not fully conformed");
        if (manifest.bundle.schema !== "v6")
            throw new Error("current oracle must select immutable v6");
    },
);
