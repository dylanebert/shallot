import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";

test("Cloudflare personal oracle preserves the terminal for inline login and prompts", () => {
    const source = readFileSync(new URL("./publish.oracle.ts", import.meta.url), "utf8");
    const firstPublish = source.match(/const first = Bun\.spawnSync\(\{([\s\S]*?)\}\);/)?.[1];
    expect(firstPublish).toBeDefined();
    expect(firstPublish).toContain('...bunChild([projectCli, "publish"])');
    expect(firstPublish).toContain('stdin: "inherit"');
    expect(firstPublish).toContain('stdout: "inherit"');
});
