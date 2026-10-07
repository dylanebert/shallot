import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
    KERNEL_ARTIFACTS,
    KERNEL_BUILD_COMMAND,
    kernelInputHash,
} from "../crates/physics/scripts/kernel-inputs";

const root = resolve(import.meta.dir, "..");
const stamp = `// Build inputs: sha256:${kernelInputHash(root)}`;
let failed = false;
for (const artifact of KERNEL_ARTIFACTS) {
    const contents = readFileSync(resolve(root, artifact), "utf8");
    if (contents.split("\n", 3)[1] === stamp) continue;
    console.error(
        `${artifact}: build-input stamp missing or stale. Regenerate: ${KERNEL_BUILD_COMMAND}`,
    );
    failed = true;
}
if (failed) process.exit(1);
