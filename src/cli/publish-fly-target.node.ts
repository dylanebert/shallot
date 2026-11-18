import { afterAll, beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { removeTestDirectory } from "../../scripts/test-directory";
import { readPublishLink } from "./publish";
import { deployFlyServer, ensureFly, flyOrganizations } from "./publish-fly";

setDefaultTimeout(TIMEOUT.node);
const root = mkdtempSync(join(tmpdir(), "shallot-fly-target-"));
const cli = join(root, process.platform === "win32" ? "fly.exe" : "fly");
const guest = { ["cpu_kind"]: "shared", cpus: 1, ["memory_mb"]: 256 };
const machines = [{ config: { guest } }];
const ips = [{ Type: "shared_v4" }, { Type: "v6" }];
const commands = join(root, "commands.jsonl");
beforeAll(async () => {
    const source = join(root, "fake.ts");
    writeFileSync(
        source,
        `import {appendFileSync,readFileSync} from 'node:fs';
import {join,dirname} from 'node:path';
const root=dirname(process.execPath);const args=process.argv.slice(2);
appendFileSync(join(root,'commands.jsonl'),JSON.stringify(args)+'\\n');
if(args[0]==='apps'&&args[1]==='list')console.log(JSON.stringify([{Name:'occupied'}]));
else if(args[0]==='machine'&&args[1]==='list')console.log(readFileSync(join(root,'machines.json'),'utf8'));
else if(args[0]==='ips'&&args[1]==='list')console.log(readFileSync(join(root,'ips.json'),'utf8'));
else if(args[0]==='orgs')console.log('{"personal":"Personal"}');`,
    );
    writeFileSync(join(root, "server.ts"), "export const state = 1;");
    writeFileSync(join(root, "machines.json"), JSON.stringify(machines));
    writeFileSync(join(root, "ips.json"), JSON.stringify(ips));
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

test("an authorized existing app inside the envelope deploys without allocating or releasing resources", async () => {
    const token = process.env.FLY_API_TOKEN;
    process.env.FLY_API_TOKEN = "fixture";
    const work = mkdtempSync(join(tmpdir(), "shallot-fly-existing-"));
    writeFileSync(commands, "");
    try {
        await deployFlyServer(
            root,
            work,
            { target: "fly", app: "occupied" },
            false,
            true,
            ask,
            cli,
        );
        const rows: string[][] = readFileSync(commands, "utf8")
            .trim()
            .split("\n")
            .map((row) => JSON.parse(row));
        expect(rows.map((row) => row.slice(0, 2).join(" "))).toEqual([
            "apps list",
            "machine list",
            "ips list",
            "deploy --app",
        ]);
    } finally {
        await removeTestDirectory(work);
        if (token === undefined) delete process.env.FLY_API_TOKEN;
        else process.env.FLY_API_TOKEN = token;
    }
});

for (const [label, inventory, addresses, reason, inspect] of [
    ["two Machines", [...machines, ...machines], ips, "Machine count 2", "machine list"],
    ["zero Machines", [], ips, "Machine count 0", "machine list"],
    [
        "dedicated CPU",
        [{ config: { guest: { ...guest, ["cpu_kind"]: "performance" } } }],
        ips,
        "Machine size",
        "machine list",
    ],
    [
        "extra CPU",
        [{ config: { guest: { ...guest, cpus: 2 } } }],
        ips,
        "Machine size",
        "machine list",
    ],
    [
        "extra memory",
        [{ config: { guest: { ...guest, ["memory_mb"]: 512 } } }],
        ips,
        "Machine size",
        "machine list",
    ],
    ["GPU", [{ config: { guest: { ...guest, gpus: 1 } } }], ips, "Machine size", "machine list"],
    [
        "GPU kind",
        [{ config: { guest: { ...guest, ["gpu_kind"]: "a100" } } }],
        ips,
        "Machine size",
        "machine list",
    ],
    ["missing Machine config", [{}], ips, "Machine size", "machine list"],
    ["invalid Machine inventory", {}, ips, "unrecognized Machine inventory", "machine list"],
    ["invalid IP inventory", machines, {}, "unrecognized IP inventory", "ips list"],
    ["dedicated IPv4", machines, [...ips, { Type: "v4" }], "IP inventory contains v4", "ips list"],
    [
        "paid egress IP",
        machines,
        [...ips, { Type: "egress_v4" }],
        "IP inventory contains egress_v4",
        "ips list",
    ],
    [
        "paid egress IPv6",
        machines,
        [...ips, { Type: "egress_v6" }],
        "IP inventory contains egress_v6",
        "ips list",
    ],
    [
        "no shared IPv4",
        machines,
        [{ Type: "v6" }],
        "IP inventory must contain one shared IPv4",
        "ips list",
    ],
] as const) {
    test(`authorized existing app refuses ${label} before any build or provider mutation`, async () => {
        const token = process.env.FLY_API_TOKEN;
        process.env.FLY_API_TOKEN = "fixture";
        const work = mkdtempSync(join(tmpdir(), "shallot-fly-refuse-"));
        writeFileSync(commands, "");
        writeFileSync(join(root, "machines.json"), JSON.stringify(inventory));
        writeFileSync(join(root, "ips.json"), JSON.stringify(addresses));
        try {
            const result = deployFlyServer(
                root,
                work,
                { target: "fly", app: "occupied" },
                false,
                true,
                ask,
                cli,
            );
            await expect(result).rejects.toThrow(reason);
            await expect(result).rejects.toThrow(`${inspect} --app occupied --json`);
            expect(
                readFileSync(commands, "utf8")
                    .trim()
                    .split("\n")
                    .map((row) => JSON.parse(row))
                    .every(
                        (row) => ["apps", "machine", "ips"].includes(row[0]) && row[1] === "list",
                    ),
            ).toBe(true);
            expect(existsSync(join(work, "fly.toml"))).toBe(false);
        } finally {
            writeFileSync(join(root, "machines.json"), JSON.stringify(machines));
            writeFileSync(join(root, "ips.json"), JSON.stringify(ips));
            await removeTestDirectory(work);
            if (token === undefined) delete process.env.FLY_API_TOKEN;
            else process.env.FLY_API_TOKEN = token;
        }
    });
}

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
    expect(() => readPublishLink(root)).toThrow("Invalid shallot.publish");
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
