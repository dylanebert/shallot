import { expect, spyOn, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Browser } from "playwright";
import * as build from "../project";
import { publishProject } from "./publish";
import * as cloudflare from "./publish-cloudflare";
import * as itch from "./publish-itch";
import * as verify from "./publish-verify";

for (const scenario of [
    "cloudflare",
    "itch",
    "itch-pending",
    "wrangler.json",
    "wrangler.jsonc",
    "wrangler.toml",
] as const) {
    const target = scenario.startsWith("itch") ? "itch" : "cloudflare";
    const pending = scenario === "itch-pending";
    const conflict = scenario.startsWith("wrangler.");
    for (const fails of pending ? [false] : [false, true]) {
        test(`${scenario} report: ${fails ? "verification failure withholds stdout" : "stdout is only the final play link"}`, async () => {
            const root = mkdtempSync(join(tmpdir(), "shallot-report-"));
            const directory = join(root, "project");
            mkdirSync(directory);
            const ancestor = join(root, scenario);
            if (conflict) writeFileSync(ancestor, `{"account_id":"${"b".repeat(32)}"}`);
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
                spyOn(itch, "runButler").mockImplementation((_butler, args) => {
                    const recorded = JSON.parse(
                        readFileSync(join(import.meta.dir, "fixtures/publish-itch.json"), "utf8"),
                    );
                    return {
                        ...recorded[args[0] === "push" ? "pushCreated" : "statusCompleted"],
                        stderr: "",
                    };
                }),
                spyOn(globalThis, "fetch").mockResolvedValue(
                    new Response(null, { status: pending ? 404 : 200 }),
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
                    if (!conflict) {
                        expect(report).toContain("Preview: ");
                        expect(report).toContain("Unpublish: ");
                    }
                    if (target === "cloudflare") {
                        expect(report).toContain("Confirm with the person before deleting");
                        if (conflict) {
                            expect(report).not.toContain("bunx wrangler");
                            expect(report).toContain(`Wrangler config found: ${ancestor}`);
                            expect(report).toContain(
                                "Wrangler would take its account_id over the link's account",
                            );
                        } else
                            for (const command of [
                                "dev --assets <web-build-directory>",
                                `rollback --name ${name}`,
                                `delete --name ${name}`,
                            ])
                                expect(report).toContain(
                                    `CLOUDFLARE_ACCOUNT_ID=${accountId} bunx wrangler ${command}`,
                                );
                        expect(report).toContain(
                            `Workers (all in this account): https://dash.cloudflare.com/${accountId}/workers/overview`,
                        );
                    } else {
                        expect(report).toContain(
                            "Confirm with the person before taking the page down",
                        );
                        expect(report).toContain('Login: "fixture-butler" login');
                        expect(report).toContain(`Status: "fixture-butler" status ${name}`);
                        expect(report).toContain(
                            "Rollback: check out an earlier commit, then bunx shallot publish --target itch",
                        );
                        expect(report).toContain(
                            "Unpublish: https://itch.io/dashboard, Edit game, set Visibility to Draft",
                        );
                    }
                    if (pending) {
                        expect(report).toContain(
                            `Not verified yet: itch.io serves this build once its page is viewed. Open ${url}, then run shallot publish again to verify https://html.itch.zone/html/19938966-2182067/index.html.`,
                        );
                        expect(verify.verifyPublishedGame).not.toHaveBeenCalled();
                        expect(report).not.toContain("Engine started");
                    } else {
                        expect(report).toContain("Engine started on a hardware adapter.");
                        expect(verify.verifyPublishedGame).toHaveBeenCalledTimes(1);
                    }
                    expect(report).not.toContain("bulk provider output");
                    expect(report).not.toContain("provider chatter");
                }
                if (fails) expect(seen.filter((entry) => entry.stream === "stdout")).toEqual([]);
            } finally {
                for (const mock of mocks) mock.mockRestore();
                process.env = env;
                rmSync(root, { recursive: true, force: true });
            }
        });
    }
}
