import { afterAll, afterEach, beforeEach, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expectError } from "../../scripts/test-console";
import { publishProject } from "./publish";
import { projectChromium } from "./publish-verify";

const root = mkdtempSync(join(tmpdir(), "shallot-publish-refusals-"));
afterAll(() => rmSync(root, { recursive: true, force: true }));
let previousCI: string | undefined;
beforeEach(() => {
    previousCI = process.env.CI;
    process.env.CI = "true";
});
afterEach(() => {
    if (previousCI === undefined) delete process.env.CI;
    else process.env.CI = previousCI;
});
const account = { id: "a".repeat(32), name: "fixture" };
const identity = { loggedIn: true, accounts: [account] };
interface Answer {
    identity?: unknown;
    versions?: { status: number; stdout: string; code?: number };
    version?: string;
    name?: string;
    link?: unknown;
}

function fixture(answer: Answer): { project: string; original: string; commands: string } {
    const project = mkdtempSync(join(root, "game-"));
    const provider = join(project, "node_modules/wrangler");
    mkdirSync(provider, { recursive: true });
    const commands = join(provider, "commands.jsonl");
    const original = JSON.stringify({
        name: answer.name ?? "fixture-game",
        ...(answer.link ? { shallot: { publish: answer.link } } : {}),
    });
    writeFileSync(join(project, "package.json"), original);
    writeFileSync(
        join(provider, "package.json"),
        JSON.stringify({
            name: "wrangler",
            version: answer.version ?? "4.151.3",
            bin: { wrangler: "cli.cjs" },
        }),
    );
    writeFileSync(
        join(provider, "answers.json"),
        JSON.stringify({ ...answer, identity: answer.identity ?? identity }),
    );
    writeFileSync(
        join(provider, "cli.cjs"),
        `
const fs = require('node:fs'), path = require('node:path');
const answer = JSON.parse(fs.readFileSync(path.join(__dirname, 'answers.json')));
const args = process.argv.slice(2);
fs.appendFileSync(path.join(__dirname, 'commands.jsonl'), JSON.stringify(args) + '\\n');
if (args[0] === 'whoami') {
  console.log(JSON.stringify(answer.identity));
  process.exit(answer.identity.loggedIn === false ? 1 : 0);
}
if (args[0] === 'versions') {
  const result = answer.versions;
  if (result.status !== 0) fs.appendFileSync(process.env.WRANGLER_OUTPUT_FILE_PATH,
    JSON.stringify({ type: 'command-failed', version: 1, code: result.code }) + '\\n');
  process.stdout.write(result.stdout);
  process.exit(result.status);
}
throw new Error('Unexpected provider mutation or prompt: ' + args.join(' '));
`,
    );
    return { project, original, commands };
}

for (const [claim, answer, message] of [
    [
        "missing login names the CI credential",
        { identity: { loggedIn: false } },
        "CLOUDFLARE_API_TOKEN",
    ],
    [
        "missing account access refuses",
        { identity: { loggedIn: true, accounts: [] } },
        "no accessible Cloudflare accounts",
    ],
    [
        "ambiguous accounts require an explicit account",
        {
            identity: {
                loggedIn: true,
                accounts: [account, { id: "b".repeat(32), name: "second" }],
            },
        },
        "CLOUDFLARE_ACCOUNT_ID",
    ],
    [
        "a saved inaccessible account refuses",
        { link: { target: "cloudflare", name: "fixture-game", accountId: "b".repeat(32) } },
        "not accessible",
    ],
    ["an invalid default name refuses", { name: "" }, "Invalid Worker name"],
    [
        "a different installed provider version refuses",
        { version: "4.151.2" },
        "requires wrangler 4.151.3",
    ],
    [
        "a taken Worker never gets a first link",
        { versions: { status: 0, stdout: "[]" } },
        'Worker "fixture-game" already exists',
    ],
    [
        "permission failure never authorizes creation",
        { versions: { status: 1, stdout: "", code: 9109 } },
        "Could not check Worker ownership",
    ],
    [
        "unstructured missing-worker prose never authorizes creation",
        { versions: { status: 1, stdout: "Worker not found [code: 10007]" } },
        "Could not check Worker ownership",
    ],
] satisfies [string, Answer, string][]) {
    test(`non-interactive publish: ${claim}`, async () => {
        const { project, original, commands } = fixture(answer);
        if (!answer.link) expectError(/^Using publish target cloudflare\.$/);
        expectError(/^Using Cloudflare account fixture /, { optional: true });
        expectError(/^Using project name .* as the Worker name\.$/, { optional: true });
        const result = publishProject(project);
        await expect(result).rejects.toThrow(message);
        expect(readFileSync(join(project, "package.json"), "utf8")).toBe(original);
        if (answer.versions?.status === 0) {
            await expect(result).rejects.toThrow("shallot.publish.name");
            expect(readFileSync(commands, "utf8")).not.toContain('"deploy"');
            expect(readFileSync(commands, "utf8")).not.toContain('"login"');
        }
    });
}

test("an unknown target is refused before reading or changing the project", async () => {
    await expect(publishProject("/no-such-project", "other")).rejects.toThrow(
        "Unknown publish target",
    );
});

test("a deliberate saved link skips the first-link collision query", async () => {
    const { project, original, commands } = fixture({
        link: { target: "cloudflare", name: "fixture-game" },
    });
    expectError(/^Using Cloudflare account fixture /);
    expectError(/^Cannot resolve Vite /);
    await expect(publishProject(project)).rejects.toThrow("vite build exited");
    expect(readFileSync(commands, "utf8")).not.toContain('"versions"');
    expect(readFileSync(join(project, "package.json"), "utf8")).toBe(original);
});

test("a saved account cannot be redirected through the environment", async () => {
    const previous = process.env.CLOUDFLARE_ACCOUNT_ID;
    process.env.CLOUDFLARE_ACCOUNT_ID = "b".repeat(32);
    try {
        const { project, original } = fixture({
            link: { target: "cloudflare", name: "fixture-game", accountId: account.id },
        });
        await expect(publishProject(project)).rejects.toThrow(
            "CLOUDFLARE_ACCOUNT_ID differs from shallot.publish.accountId",
        );
        expect(readFileSync(join(project, "package.json"), "utf8")).toBe(original);
    } finally {
        if (previous === undefined) delete process.env.CLOUDFLARE_ACCOUNT_ID;
        else process.env.CLOUDFLARE_ACCOUNT_ID = previous;
    }
});

test("a missing project Playwright names its install command", async () => {
    const project = mkdtempSync(join(root, "no-browser-"));
    writeFileSync(join(project, "package.json"), "{}");
    await expect(projectChromium(project)).rejects.toThrow("bun add --dev playwright@");
});

test("missing Chromium installs through the project's declared Playwright binary", async () => {
    const project = mkdtempSync(join(root, "browser-"));
    const provider = join(project, "node_modules/playwright");
    mkdirSync(provider, { recursive: true });
    writeFileSync(join(project, "package.json"), "{}");
    writeFileSync(
        join(provider, "package.json"),
        JSON.stringify({
            name: "playwright",
            type: "module",
            exports: { ".": "./index.js", "./package.json": "./package.json" },
            bin: { playwright: "cli.js" },
        }),
    );
    writeFileSync(
        join(provider, "index.js"),
        `
let count = 0;
export const chromium = { async launch() {
    if (count++ === 0) throw new Error("Executable doesn't exist");
    return { async close() {} };
} };
`,
    );
    writeFileSync(
        join(provider, "cli.js"),
        `
import { writeFileSync } from 'node:fs';
writeFileSync(new URL('./installed.json', import.meta.url), JSON.stringify(process.argv.slice(2)));
`,
    );
    expectError(/^Installing the project's Playwright Chromium\.$/);
    const browser = await projectChromium(project);
    expect(JSON.parse(readFileSync(join(provider, "installed.json"), "utf8"))).toEqual([
        "install",
        "chromium",
    ]);
    await browser.close();
});
