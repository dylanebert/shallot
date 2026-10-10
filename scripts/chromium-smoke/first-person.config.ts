import { fileURLToPath } from "node:url";
import base from "../../examples/first-person/playwright.config";
import { chromiumVariant } from "./variants";

const selected = process.env.PORTABLE_WEBGPU_VARIANT;
if (!selected) throw new Error("PORTABLE_WEBGPU_VARIANT must name a tested stable variant");

export default {
    ...base,
    testDir: fileURLToPath(new URL("../../examples/first-person/", import.meta.url)),
    use: {
        ...base.use,
        launchOptions: {
            ...base.use.launchOptions,
            args: [...chromiumVariant(selected).args],
        },
    },
};
