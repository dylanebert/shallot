import { expect, spyOn, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Browser } from "playwright";
import * as build from "../project";
import { publishProject } from "./publish";
import * as cloudflare from "./publish-cloudflare";
import * as itch from "./publish-itch";
import * as verify from "./publish-verify";

for (const target of ["cloudflare", "itch"] as const) {
    for (const fails of [false, true]) {
        test(`${target} report: ${fails ? "verification failure withholds stdout" : "stdout is only the final play link"}`, async () => {
            const directory = mkdtempSync(join(tmpdir(), "shallot-report-"));
            const accountId = "a".repeat(32);
            const name = target === "itch" ? "user/game" : "game";
            const url =
                target === "itch" ? "https://user.itch.io/game" : "https://game.user.workers.dev";
            writeFileSync(
                join(directory, "package.json"),
                JSON.stringify({
                    shallot: {
                        publish: {
                            target,
                            name,
                            ...(target === "cloudflare" ? { accountId } : {}),
                        },
                    },
                }),
            );
            const env = { ...process.env };
            process.env.CI = "true";
            process.env.CLOUDFLARE_ACCOUNT_ID = accountId;
            process.env.CLOUDFLARE_API_TOKEN = "fixture";
            process.env.BUTLER_API_KEY = "fixture";
            const seen: { stream: string; text: string }[] = [];
            const mocks = [
                spyOn(console, "log").mockImplementation((text) => {
                    seen.push({ stream: "stdout", text: `${text}\n` });
                }),
                spyOn(process.stderr, "write").mockImplementation((text) => {
                    seen.push({ stream: "stderr", text: String(text) });
                    return true;
                }),
                spyOn(build, "buildWeb").mockImplementation(async (_project, options) => {
                    expect(options?.stdout).toBe("quiet");
                    mkdirSync(options!.outDir!, { recursive: true });
                    writeFileSync(join(options!.outDir!, "index.html"), "game");
                }),
                spyOn(cloudflare, "resolveWrangler").mockReturnValue("fixture"),
                spyOn(cloudflare, "runWrangler").mockReturnValue({
                    status: 0,
                    stdout: "bulk provider output",
                    stderr: "provider chatter",
                    entries: [
                        {
                            type: "deploy",
                            ...JSON.parse('{"version_id":"12345678-rest"}'),
                            targets: [url],
                        },
                    ],
                }),
                spyOn(itch, "ensureButler").mockResolvedValue("fixture-butler"),
                spyOn(itch, "runButler").mockReturnValue({ status: 0, stderr: "", entries: [] }),
                spyOn(itch, "itchChannel").mockReturnValue({ uploadId: 1 }),
                spyOn(itch, "itchPush").mockReturnValue(2),
                spyOn(itch, "waitForItchBuild").mockResolvedValue({ uploadId: 1, buildId: 2 }),
                spyOn(itch, "liveItchBuild").mockResolvedValue(
                    "https://html.itch.zone/html/1-2/index.html",
                ),
                spyOn(verify, "projectChromium").mockResolvedValue({
                    close: async () => {
                        seen.push({ stream: "stderr", text: "browser closed\n" });
                    },
                } as Browser),
                spyOn(verify, "verifyPublishedGame").mockImplementation(async () => {
                    if (fails) throw new Error("verification failed");
                    return {
                        hardware: true,
                        adapter: "fixture adapter",
                    } as verify.PublishVerification;
                }),
            ];
            try {
                if (fails)
                    await expect(publishProject(directory)).rejects.toThrow("verification failed");
                else {
                    await publishProject(directory);
                    expect(seen.filter((entry) => entry.stream === "stdout")).toEqual([
                        { stream: "stdout", text: `${url}\n` },
                    ]);
                    expect(seen.at(-1)).toEqual({ stream: "stdout", text: `${url}\n` });
                    const report = seen.map((entry) => entry.text).join("");
                    expect(report).toContain("Build: ");
                    expect(report).toContain("Preview: ");
                    expect(report).toContain("Unpublish: ");
                    expect(report).toContain("Engine started on a hardware adapter.");
                    expect(report).not.toContain("bulk provider output");
                    expect(report).not.toContain("provider chatter");
                }
                if (fails) expect(seen.filter((entry) => entry.stream === "stdout")).toEqual([]);
            } finally {
                for (const mock of mocks) mock.mockRestore();
                process.env = env;
                rmSync(directory, { recursive: true, force: true });
            }
        });
    }
}
