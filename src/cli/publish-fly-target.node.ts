import { afterAll, beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { removeTestDirectory } from "../../scripts/test-directory";
import { readPublishLink } from "./publish";
import { deployFlyServer, ensureFly, flyOrganizations } from "./publish-fly";

setDefaultTimeout(TIMEOUT.node);
const root = mkdtempSync(join(tmpdir(), "shallot-fly-target-"));
const cli = join(root, process.platform === "win32" ? "fly.exe" : "fly");
beforeAll(async () => {
    const source = join(root, "fake.ts");
    writeFileSync(
        source,
        `if(process.argv[2]==='apps'&&process.argv[3]==='list')console.log(JSON.stringify([{Name:'occupied'}]));else if(process.argv[2]==='orgs')console.log('{"personal":"Personal"}');`,
    );
    writeFileSync(join(root, "server.ts"), "export const state = 1;");
    const result = await Bun.build({ entrypoints: [source], compile: { outfile: cli } });
    if (!result.success) throw new Error(result.logs.join("\n"));
});
afterAll(() => removeTestDirectory(root));
const ask = async () => {
    throw new Error("Unexpected prompt");
};

test("Fly organizations use the pinned CLI's slug-to-name JSON object", () => {
    expect(flyOrganizations('{"personal":"Personal"}')).toEqual(["personal"]);
    expect(flyOrganizations('{"personal":"Personal","studio":"Studio"}')).toEqual([
        "personal",
        "studio",
    ]);
    for (const invalid of ["[]", "{}", "null", '{"personal":false}'])
        expect(() => flyOrganizations(invalid)).toThrow("organizations");
});

test("an absent app deploys with the sole organization without prompting", async () => {
    const token = process.env.FLY_API_TOKEN;
    process.env.FLY_API_TOKEN = "fixture";
    const work = mkdtempSync(join(tmpdir(), "shallot-fly-work-"));
    try {
        await deployFlyServer(
            root,
            work,
            { target: "fly", app: "scratch-app" },
            false,
            false,
            ask,
            cli,
        );
        expect(readFileSync(join(work, "fly.toml"), "utf8")).toContain('app = "scratch-app"');
    } finally {
        await removeTestDirectory(work);
        if (token === undefined) delete process.env.FLY_API_TOKEN;
        else process.env.FLY_API_TOKEN = token;
    }
});

test("server links retain Fly and reject malformed targets and app names", () => {
    const path = join(root, "package.json");
    const server = { target: "fly", app: "scratch-app" } as const;
    const link = { target: "cloudflare", name: "game", accountId: "a".repeat(32), server } as const;
    writeFileSync(path, JSON.stringify({ shallot: { publish: link } }));
    expect(readPublishLink(root)).toEqual(link);
    for (const invalid of [
        { target: "other", app: "scratch-app" },
        { target: "fly", app: "../bad" },
        null,
    ]) {
        writeFileSync(path, JSON.stringify({ shallot: { publish: { ...link, server: invalid } } }));
        expect(() => readPublishLink(root)).toThrow("shallot.publish.server");
    }
    writeFileSync(
        path,
        JSON.stringify({ shallot: { publish: { target: "itch", name: "user/game", server } } }),
    );
    expect(readPublishLink(root)?.server).toEqual(server);
    writeFileSync(path, JSON.stringify({ shallot: { publish: { server } } }));
    expect(readPublishLink(root)).toEqual({ server });
});

test("a first Fly link refuses an existing app without building or changing project files", async () => {
    const token = process.env.FLY_API_TOKEN;
    process.env.FLY_API_TOKEN = "fixture";
    try {
        const before = readFileSync(join(root, "package.json"), "utf8");
        await expect(
            deployFlyServer(root, root, { target: "fly", app: "occupied" }, false, false, ask, cli),
        ).rejects.toThrow("already exists");
        expect(readFileSync(join(root, "package.json"), "utf8")).toBe(before);
    } finally {
        if (token === undefined) delete process.env.FLY_API_TOKEN;
        else process.env.FLY_API_TOKEN = token;
    }
});

test("noninteractive Fly login requires an explicit token, not a saved login", async () => {
    const token = process.env.FLY_API_TOKEN;
    delete process.env.FLY_API_TOKEN;
    try {
        await expect(
            deployFlyServer(
                root,
                root,
                { target: "fly", app: "scratch-app" },
                false,
                false,
                ask,
                cli,
            ),
        ).rejects.toThrow("FLY_API_TOKEN");
    } finally {
        if (token !== undefined) process.env.FLY_API_TOKEN = token;
    }
});

test("project Fly configuration refuses before login or remote commands", async () => {
    const project = mkdtempSync(join(tmpdir(), "shallot-fly-config-"));
    try {
        writeFileSync(join(project, "fly.toml"), "creator config");
        await expect(
            deployFlyServer(
                project,
                root,
                { target: "fly", app: "scratch-app" },
                false,
                false,
                ask,
                "must-not-run",
            ),
        ).rejects.toThrow("Fly config found");
    } finally {
        await removeTestDirectory(project);
    }
});

test("a flyctl archive with the wrong release hash never extracts or runs", async () => {
    const cache = process.env.XDG_CACHE_HOME;
    process.env.XDG_CACHE_HOME = join(root, "cache");
    try {
        await expect(
            ensureFly("Linux_x86_64", "data:application/octet-stream,not-flyctl"),
        ).rejects.toThrow("SHA-256");
    } finally {
        if (cache === undefined) delete process.env.XDG_CACHE_HOME;
        else process.env.XDG_CACHE_HOME = cache;
    }
});
