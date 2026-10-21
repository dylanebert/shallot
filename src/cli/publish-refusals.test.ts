import { afterAll, afterEach, beforeEach, expect, spyOn, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, relative } from "node:path";
import { expectError } from "../../scripts/test-console";
import { normalizePath } from "../project";
import { publishHeaders, publishProject, readPublishLink } from "./publish";
import { projectChromium } from "./publish-verify";

const root = mkdtempSync(join(tmpdir(), "shallot-publish-refusals-"));
afterAll(() => rmSync(root, { recursive: true, force: true }));
const progress: string[] = [];
const expectedProgress: { pattern: RegExp; optional?: boolean }[] = [];
let stderr: ReturnType<typeof spyOn<typeof process.stderr, "write">>;
function expectProgress(pattern: RegExp, options: { optional?: boolean } = {}): void {
    expectedProgress.push({ pattern, ...options });
}
let previousCI: string | undefined;
let previousAccount: string | undefined;
let previousToken: string | undefined;
beforeEach(() => {
    progress.length = 0;
    expectedProgress.length = 0;
    stderr = spyOn(process.stderr, "write").mockImplementation((chunk) => {
        progress.push(chunk.toString().replace(/\n$/, ""));
        return true;
    });
    previousCI = process.env.CI;
    previousAccount = process.env.CLOUDFLARE_ACCOUNT_ID;
    previousToken = process.env.CLOUDFLARE_API_TOKEN;
    delete process.env.CLOUDFLARE_API_TOKEN;
    process.env.CI = "true";
    delete process.env.CLOUDFLARE_ACCOUNT_ID;
});
afterEach(() => {
    if (previousCI === undefined) delete process.env.CI;
    else process.env.CI = previousCI;
    if (previousAccount === undefined) delete process.env.CLOUDFLARE_ACCOUNT_ID;
    else process.env.CLOUDFLARE_ACCOUNT_ID = previousAccount;
    if (previousToken === undefined) delete process.env.CLOUDFLARE_API_TOKEN;
    else process.env.CLOUDFLARE_API_TOKEN = previousToken;
    stderr.mockRestore();
    for (const { pattern, optional } of expectedProgress) {
        if (!optional)
            expect(
                progress.some((message) => pattern.test(message)),
                `Missing stderr matching ${pattern}`,
            ).toBe(true);
    }
});
const account = { id: "a".repeat(32), name: "fixture" };
const identity = { loggedIn: true, accounts: [account] };
interface Answer {
    identity?: unknown;
    whoamiFailure?: string;
    versions?: { status: number; stdout: string; code?: number };
    version?: string;
    name?: string;
    link?: unknown;
    captureDeploy?: boolean;
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
  if (answer.whoamiFailure) {
    console.error(answer.whoamiFailure);
    process.exit(1);
  }
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
if (args[0] === 'deploy' && answer.captureDeploy) {
  const configPath = args[args.indexOf('--config') + 1];
  const config = JSON.parse(fs.readFileSync(configPath));
  fs.writeFileSync(path.join(__dirname, 'deployment.json'), JSON.stringify({
    cwd: process.cwd(), configPath, config,
    headers: fs.readFileSync(path.join(config.assets.directory, '_headers'), 'utf8')
  }));
  console.error('fixture deploy refused');
  process.exit(1);
}
throw new Error('Unexpected provider mutation or prompt: ' + args.join(' '));
`,
    );
    if (answer.versions) {
        const vite = join(project, "node_modules/vite");
        mkdirSync(vite, { recursive: true });
        writeFileSync(
            join(vite, "package.json"),
            JSON.stringify({ name: "vite", bin: { vite: "cli.cjs" } }),
        );
        writeFileSync(
            join(vite, "cli.cjs"),
            `const fs = require('node:fs'), path = require('node:path'); const directory = process.argv[process.argv.indexOf('--outDir') + 1]; fs.mkdirSync(directory, {recursive:true}); fs.writeFileSync(path.join(directory,'index.html'), '<title>fixture</title>');`,
        );
    }
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
        expectProgress(/^Using publish target cloudflare\.$/);
        expectProgress(/^Using Cloudflare account fixture /, { optional: true });
        expectProgress(/^Using project name .* as the Worker name\.$/, { optional: true });
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

test("first-link setup builds and writes the target's artifact before invoking deploy", async () => {
    const { project, commands } = fixture({
        captureDeploy: true,
        versions: { status: 1, stdout: "", code: 10007 },
    });
    expectProgress(/^Using publish target cloudflare\.$/);
    expectProgress(/^Using Cloudflare account fixture /);
    expectProgress(/^Using project name fixture-game as the Worker name\.$/);
    expectProgress(/^\s*fixture deploy refused\n$/);
    await expect(publishProject(project)).rejects.toThrow("Wrangler deploy failed");
    const deployment = JSON.parse(
        readFileSync(join(project, "node_modules/wrangler/deployment.json"), "utf8"),
    );
    expect(realpathSync(deployment.cwd)).toBe(realpathSync(project));
    expect(normalizePath(relative(project, deployment.configPath)).startsWith("..")).toBe(true);
    expect(deployment.config.assets.not_found_handling).toBe("none");
    expect(deployment.headers).toBe(publishHeaders());
    expect(readPublishLink(project)).toBeUndefined();
    expect(
        readFileSync(commands, "utf8")
            .trim()
            .split("\n")
            .map((line) => JSON.parse(line)[0]),
    ).toEqual(["whoami", "versions", "deploy"]);
});

test("a deliberate saved link skips the first-link collision query", async () => {
    const { project, original, commands } = fixture({
        link: { target: "cloudflare", name: "fixture-game", accountId: account.id },
    });
    expectProgress(/^Using Cloudflare account fixture /);
    expectError(/^Cannot resolve Vite /);
    await expect(publishProject(project)).rejects.toThrow("vite build exited");
    expect(readFileSync(commands, "utf8")).not.toContain('"versions"');
    expect(readFileSync(join(project, "package.json"), "utf8")).toBe(original);
});

// Recorded on Actions run 39684554804 with Wrangler 4.151.3 and an account-scoped token.
const whoamiFailure = "Failed to automatically retrieve account IDs for the logged in user.";

for (const source of ["environment", "saved link"] as const) {
    test(`an explicit account from ${source} does not require whoami account access`, async () => {
        if (source === "environment") process.env.CLOUDFLARE_ACCOUNT_ID = account.id;
        const { project, original, commands } = fixture({
            whoamiFailure,
            ...(source === "saved link"
                ? {
                      link: { target: "cloudflare", name: "fixture-game", accountId: account.id },
                      captureDeploy: true,
                  }
                : {}),
            versions: { status: 1, stdout: "", code: 9109 },
        });
        if (source === "environment") {
            expectProgress(/^Using publish target cloudflare\.$/);
            expectProgress(/^Using project name fixture-game as the Worker name\.$/);
        }
        expectProgress(/^Using Cloudflare account /);
        if (source === "saved link") expectProgress(/^\s*fixture deploy refused\n$/);
        await expect(publishProject(project)).rejects.toThrow(
            source === "environment"
                ? "Could not check Worker ownership"
                : "Wrangler deploy failed",
        );
        expect(readFileSync(join(project, "package.json"), "utf8")).toBe(original);
        expect(readFileSync(commands, "utf8")).toContain('"whoami"');
        if (source === "environment")
            expect(readFileSync(commands, "utf8")).not.toContain('"deploy"');
    });
}

test("a token and explicit account bypass account enumeration, but not ownership errors", async () => {
    process.env.CLOUDFLARE_API_TOKEN = "fixture-token";
    process.env.CLOUDFLARE_ACCOUNT_ID = account.id;
    const { project, commands } = fixture({
        whoamiFailure,
        versions: { status: 1, stdout: "", code: 9109 },
    });
    expectProgress(/^Using publish target cloudflare\.$/);
    expectProgress(/^Using Cloudflare account /);
    expectProgress(/^Using project name fixture-game as the Worker name\.$/);
    await expect(publishProject(project)).rejects.toThrow("Could not check Worker ownership");
    const recorded = readFileSync(commands, "utf8");
    expect(recorded).not.toContain('"whoami"');
    expect(recorded).not.toContain('"deploy"');
    expect(readPublishLink(project)).toBeUndefined();
});

test("an explicit account without credentials still refuses missing login", async () => {
    process.env.CLOUDFLARE_ACCOUNT_ID = account.id;
    const { project } = fixture({ identity: { loggedIn: false } });
    expectProgress(/^Using publish target cloudflare\.$/);
    await expect(publishProject(project)).rejects.toThrow("CLOUDFLARE_API_TOKEN");
});

test("a saved account cannot be redirected through the environment", async () => {
    const previous = process.env.CLOUDFLARE_ACCOUNT_ID;
    process.env.CLOUDFLARE_ACCOUNT_ID = "b".repeat(32);
    try {
        const { project, original } = fixture({
            link: { target: "cloudflare", name: "fixture-game", accountId: account.id },
        });
        await expect(publishProject(project)).rejects.toThrow(
            `Resolved Cloudflare account ${"b".repeat(32)} differs from shallot.publish.accountId ${account.id}`,
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
    expectProgress(/^Installing the project's Playwright Chromium\.$/);
    const browser = await projectChromium(project);
    expect(JSON.parse(readFileSync(join(provider, "installed.json"), "utf8"))).toEqual([
        "install",
        "chromium",
    ]);
    await browser.close();
});
