import { afterAll, afterEach, beforeEach, expect, spyOn, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { deflateRawSync } from "node:zlib";
import { parseCliArgs } from "./index";
import { preflightPublish, publishProject, readPublishLink } from "./publish";
import {
    type ButlerResult,
    butlerPath,
    butlerPlatform,
    ensureButler,
    extractZip,
    itchBuildState,
    itchBuildUrl,
    itchChannel,
    itchPageSettings,
    itchPageUrl,
    itchPush,
    liveItchBuild,
    waitForItchBuild,
} from "./publish-itch";

const recorded = JSON.parse(
    readFileSync(join(import.meta.dir, "fixtures/publish-itch.json"), "utf8"),
) as Record<string, ButlerResult> & {};
const answer = (name: string): ButlerResult => ({ ...recorded[name]!, stderr: "" });
const project = "individualkex/shallot-publish-test";
const page = itchPageUrl(project);
const root = mkdtempSync(join(tmpdir(), "shallot-publish-itch-"));
afterAll(() => rmSync(root, { recursive: true, force: true }));
let cache: Record<string, string | undefined>;
let stderr: ReturnType<typeof spyOn<typeof process.stderr, "write">>;
beforeEach(() => {
    cache = { XDG_CACHE_HOME: process.env.XDG_CACHE_HOME, LOCALAPPDATA: process.env.LOCALAPPDATA };
    process.env.XDG_CACHE_HOME = process.env.LOCALAPPDATA = mkdtempSync(join(root, "cache-"));
    stderr = spyOn(process.stderr, "write").mockImplementation(() => true);
});
afterEach(() => {
    for (const [name, value] of Object.entries(cache)) {
        if (value === undefined) delete process.env[name];
        else process.env[name] = value;
    }
    stderr.mockRestore();
});

test("publish accepts the itch target and saves an itch link of <user>/<game>", () => {
    expect(parseCliArgs(["publish", "--target", "itch"])).toEqual({
        kind: "publish",
        target: "itch",
    });
    const directory = mkdtempSync(join(root, "link-"));
    const write = (link: unknown) =>
        writeFileSync(
            join(directory, "package.json"),
            JSON.stringify({ shallot: { publish: link } }),
        );
    write({ target: "itch", name: project });
    expect(readPublishLink(directory)).toEqual({ target: "itch", name: project });
    for (const invalid of ["shallot-publish-test", "a/b/c", "a/", "a b/c"]) {
        write({ target: "itch", name: invalid });
        expect(() => readPublishLink(directory)).toThrow("Invalid shallot.publish");
    }
    expect(page).toBe("https://individualkex.itch.io/shallot-publish-test");
});

test("preflight refuses a build over itch.io's total", () => {
    const limits = { files: 1000, bytes: 200_000_000, total: 500_000_000 };
    const assets = [
        { path: "index.html", size: 1 },
        ...[1, 2, 3].map((n) => ({ path: `assets/${n}.bin`, size: 199_000_000 })),
    ];
    expect(() => preflightPublish(assets, limits)).toThrow(
        "The web build's 597000001 bytes exceed the target's 500000000 byte (500 MB) total.",
    );
    expect(() => preflightPublish([{ path: "index.html", size: 200_000_001 }], limits)).toThrow(
        "index.html: 200000001 bytes exceeds the target's 200000000 byte (200 MB) file limit.",
    );
});

test("butler's platforms are the ones broth builds", () => {
    expect(butlerPlatform("darwin", "arm64")).toBe("darwin-arm64");
    expect(butlerPlatform("linux", "x64")).toBe("linux-amd64");
    expect(butlerPlatform("win32", "x64")).toBe("windows-amd64");
    expect(() => butlerPlatform("win32", "arm64")).toThrow("has no build for win32-arm64");
    expect(butlerPath("windows-amd64")).toEndWith(join("15.32.0", "windows-amd64", "butler.exe"));
});

function zip(files: Record<string, { data: string; mode: number }>): Buffer {
    const locals: Buffer[] = [];
    const central: Buffer[] = [];
    let offset = 0;
    for (const [name, { data, mode }] of Object.entries(files)) {
        const body = deflateRawSync(Buffer.from(data));
        const local = Buffer.alloc(30);
        local.writeUInt32LE(0x04034b50, 0);
        local.writeUInt16LE(8, 8);
        local.writeUInt32LE(body.length, 18);
        local.writeUInt16LE(name.length, 26);
        const entry = Buffer.alloc(46);
        entry.writeUInt32LE(0x02014b50, 0);
        entry.writeUInt16LE(8, 10);
        entry.writeUInt32LE(body.length, 20);
        entry.writeUInt16LE(name.length, 28);
        entry.writeUInt32LE((0o100000 | mode) * 0x10000, 38);
        entry.writeUInt32LE(offset, 42);
        locals.push(local, Buffer.from(name), body);
        central.push(entry, Buffer.from(name));
        offset += 30 + name.length + body.length;
    }
    const directory = Buffer.concat(central);
    const end = Buffer.alloc(22);
    end.writeUInt32LE(0x06054b50, 0);
    end.writeUInt16LE(central.length / 2, 10);
    end.writeUInt32LE(directory.length, 12);
    end.writeUInt32LE(offset, 16);
    return Buffer.concat([...locals, directory, end]);
}

test("butler's archive extracts with its modes, and a changed archive is refused uncached", async () => {
    const archive = zip({
        butler: { data: "#!/bin/sh\n", mode: 0o755 },
        "7z.so": { data: "library", mode: 0o644 },
    });
    const directory = mkdtempSync(join(root, "zip-"));
    extractZip(archive, directory);
    expect(readFileSync(join(directory, "butler"), "utf8")).toBe("#!/bin/sh\n");
    expect(readFileSync(join(directory, "7z.so"), "utf8")).toBe("library");
    if (process.platform !== "win32")
        expect(statSync(join(directory, "butler")).mode & 0o777).toBe(0o755);
    expect(() => extractZip(zip({ "../escape": { data: "", mode: 0o644 } }), directory)).toThrow(
        "escapes",
    );
    const server = Bun.serve({
        port: 0,
        hostname: "127.0.0.1",
        fetch: () => new Response(new Uint8Array(archive)),
    });
    try {
        await expect(ensureButler("linux-amd64", server.url.href)).rejects.toThrow(
            "not the pinned 2335971394ef6596f95ded0833e85ee28755e13761716ef4c91d6b11f69162f5; refusing to run it",
        );
        expect(existsSync(dirname(butlerPath("linux-amd64")))).toBe(false);
    } finally {
        await server.stop(true);
    }
});

test("butler's recorded answers name a missing login and a missing project", () => {
    for (const name of ["statusNoCredentials"])
        expect(() => itchChannel(project, answer(name))).toThrow(
            "itch.io login is missing. Set BUTLER_API_KEY",
        );
    for (const name of ["statusMissingGame", "statusMissingUser"])
        expect(() => itchChannel(project, answer(name))).toThrow(
            `itch.io has no project ${project} for this login. Create it at https://itch.io/game/new with Kind of project HTML`,
        );
    expect(() => itchPush(project, answer("pushMissingGame"))).toThrow(
        `itch.io has no project ${project}`,
    );
    expect(() => itchChannel(project, answer("statusInvalidKey"))).toThrow(
        "butler status failed: itch.io API error (403): /wharf/channels: invalid key",
    );
});

test("a pushed build is live only once it is the channel's completed head", () => {
    expect(itchPush(project, answer("pushCreated"))).toBe(2182067);
    expect(itchPush(project, answer("pushUnchanged"))).toBeUndefined();
    // A first build lists no channel until processed.
    expect(itchBuildState(itchChannel(project, answer("statusNoChannel")), 2182067)).toEqual({
        done: false,
        state: "queued",
    });
    for (const [name, state] of [
        ["statusQueued", "queued"],
        ["statusProcessing", "processing"],
    ] as const) {
        const channel = itchChannel(project, answer(name));
        expect(itchBuildState(channel, 2182067)).toEqual({ done: false, state });
        // An unchanged push waits for whatever is pending.
        expect(itchBuildState(channel)).toEqual({ done: false, state });
    }
    const live = { done: true as const, uploadId: 19938966, buildId: 2182067 };
    expect(itchBuildState(itchChannel(project, answer("statusCompleted")), 2182067)).toEqual(live);
    expect(itchBuildState(itchChannel(project, answer("statusCompleted")))).toEqual(live);
    const failed = structuredClone(answer("statusProcessing"));
    (failed.entries[0] as { value: { channels: { pending: { state: string } }[] } }).value
        .channels[0]!.pending.state = "failed";
    expect(() => itchBuildState(itchChannel(project, failed), 2182067)).toThrow(
        "itch.io failed to process build 2182067",
    );
});

test("the processing wait polls until the build is the completed head, and is bounded", async () => {
    const answers = ["statusNoChannel", "statusQueued", "statusProcessing", "statusCompleted"].map(
        answer,
    );
    expect(await waitForItchBuild(project, 2182067, () => answers.shift()!, 1000, 1)).toMatchObject(
        {
            uploadId: 19938966,
            buildId: 2182067,
        },
    );
    await expect(
        waitForItchBuild(project, 2182067, () => answer("statusProcessing"), 20, 5),
    ).rejects.toThrow("itch.io has not processed build 2182067 after 0 minutes (processing)");
});

test("the report names the page settings verification cannot see, and where to set them", () => {
    expect(itchBuildUrl(19938966, 2182067)).toBe(
        "https://html.itch.zone/html/19938966-2182067/index.html",
    );
    const settings = itchPageSettings(project);
    for (const part of [
        "https://itch.io/dashboard, Edit shallot-publish-test",
        '"This file will be played in the browser" on the html5 upload',
        '"SharedArrayBuffer support" under Embed options',
        "set Visibility to Public or Restricted, not Draft",
        page,
    ])
        expect(settings).toContain(part);
});

test("a processed build not yet served refuses verification, naming the page to open", async () => {
    // Recorded: build 2182534 answered 404 from completion until its page was first viewed, 23 minutes later.
    let served = false;
    const queries: string[] = [];
    const server = Bun.serve({
        port: 0,
        hostname: "127.0.0.1",
        fetch: (request) => {
            queries.push(new URL(request.url).search);
            return new Response(null, { status: served ? 200 : 404 });
        },
    });
    try {
        const build = `${server.url}html/19938966-2182534/index.html`;
        await expect(liveItchBuild(page, build)).rejects.toThrow(
            `Not verified yet: itch.io serves this build once its page is viewed. Open ${page}, then run shallot publish again to verify ${build}.`,
        );
        served = true;
        const live = await liveItchBuild(page, build);
        expect(live).toStartWith(`${build}?v=`);
        expect(queries.every((query) => query.startsWith("?v="))).toBe(true);
    } finally {
        await server.stop(true);
    }
});

test("an unlinked itch publish without a terminal refuses before fetching butler or building", async () => {
    const directory = mkdtempSync(join(root, "unlinked-"));
    writeFileSync(join(directory, "package.json"), JSON.stringify({ name: "game" }));
    const ci = process.env.CI;
    process.env.CI = "true";
    try {
        await expect(publishProject(directory, "itch")).rejects.toThrow(
            'No itch.io project is linked. Create the page at https://itch.io/game/new with Kind of project HTML; shallot.publish in package.json then names it, {"target": "itch", "name": "<user>/<game>"}.',
        );
    } finally {
        if (ci === undefined) delete process.env.CI;
        else process.env.CI = ci;
    }
    expect(existsSync(dirname(butlerPath()))).toBe(false);
    expect(JSON.parse(readFileSync(join(directory, "package.json"), "utf8"))).toEqual({
        name: "game",
    });
});
