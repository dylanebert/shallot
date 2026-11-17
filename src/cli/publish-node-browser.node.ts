import { expect, setDefaultTimeout, test } from "bun:test";
import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { removeTestDirectory } from "../../scripts/test-directory";

setDefaultTimeout(TIMEOUT.node);

test("Node resolves Playwright's CommonJS forwarding entry before launching", async () => {
    const root = mkdtempSync(join(tmpdir(), "shallot-node-browser-"));
    try {
        const module = join(root, "node_modules/playwright");
        mkdirSync(module, { recursive: true });
        writeFileSync(join(root, "package.json"), "{}");
        writeFileSync(join(module, "package.json"), JSON.stringify({ main: "index.cjs" }));
        writeFileSync(
            join(module, "index.cjs"),
            'module.exports = require("./implementation.cjs");',
        );
        writeFileSync(
            join(module, "implementation.cjs"),
            'const name="chromium"; module.exports[name]={launch:async()=>({id:"launched"})};',
        );
        const source = join(root, "runner.ts");
        writeFileSync(
            source,
            `import {projectChromium} from ${JSON.stringify(join(import.meta.dir, "publish-verify.ts"))}; console.log((await projectChromium(${JSON.stringify(root)})).id);`,
        );
        const outfile = join(root, "runner.mjs");
        const build = await Bun.build({
            entrypoints: [source],
            target: "node",
            format: "esm",
            outdir: root,
            naming: "runner.mjs",
        });
        expect(build.success, build.logs.join("\n")).toBe(true);
        const result = spawnSync("node", [outfile], {
            encoding: "utf8",
            stdio: ["ignore", "pipe", "pipe"],
        });
        expect(result.status, result.stderr).toBe(0);
        expect(result.stdout.trim()).toBe("launched");
    } finally {
        await removeTestDirectory(root);
    }
});
