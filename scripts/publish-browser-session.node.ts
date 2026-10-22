import { expect, setDefaultTimeout, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { browserSessionPath } from "./publish-browser-session";

setDefaultTimeout(TIMEOUT.node);

test("desktop opener uses the person's session while Wrangler config stays isolated", () => {
    const root = mkdtempSync(join(tmpdir(), "shallot-browser-session-"));
    try {
        const bin = join(root, "stub-bin");
        const home = join(root, "clean-home");
        mkdirSync(bin);
        mkdirSync(home);
        const output = join(root, "opened.json");
        writeFileSync(
            join(bin, "xdg-open"),
            `#!/bin/sh\nexec '${process.execPath}' -e 'await Bun.write(${JSON.stringify(output)}, JSON.stringify({home: process.env.HOME, config: process.env.XDG_CONFIG_HOME, runtime: process.env.XDG_RUNTIME_DIR, url: process.argv[1]}))' "$@"\n`,
            { mode: 0o700 },
        );
        const session: NodeJS.ProcessEnv = { ...process.env, PATH: `${bin}:${process.env.PATH}` };
        const env = {
            ...process.env,
            HOME: home,
            XDG_CONFIG_HOME: home,
            XDG_RUNTIME_DIR: join(root, "not-the-session"),
            PATH: browserSessionPath(root, session),
        };
        expect(Bun.spawnSync(["xdg-open", "https://example.invalid/login"], { env }).exitCode).toBe(
            0,
        );
        expect(JSON.parse(readFileSync(output, "utf8"))).toEqual({
            home: session.HOME,
            config: session.XDG_CONFIG_HOME,
            runtime: session.XDG_RUNTIME_DIR,
            url: "https://example.invalid/login",
        });
        const wrangler =
            process.env.SHALLOT_TEST_WRANGLER ??
            resolve(import.meta.dir, "../node_modules/wrangler/bin/wrangler.js");
        const result = Bun.spawnSync(["node", wrangler, "telemetry", "disable"], { env });
        expect(result.exitCode, result.stderr.toString()).toBe(0);
        expect(
            JSON.parse(readFileSync(join(home, ".wrangler/metrics.json"), "utf8")).permission
                .enabled,
        ).toBe(false);
    } finally {
        rmSync(root, { recursive: true, force: true });
    }
});
