import { afterAll, beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
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
const volume = {
    id: "vol_2n0l9vl2l9pr635d",
    name: "shallot_data",
    ["size_gb"]: 1,
    region: "iad",
    ["attached_machine_id"]: "7591857c12836d",
    ["auto_backup_enabled"]: false,
};
const machines = [
    {
        id: volume.attached_machine_id,
        region: volume.region,
        config: {
            guest,
            metadata: { ["fly_platform_version"]: "v2", ["fly_process_group"]: "app" },
            mounts: [{ volume: volume.id, path: "/data" }],
        },
    },
];
const ips = [{ Type: "shared_v4" }, { Type: "v6" }];
const commands = join(root, "commands.jsonl");
beforeAll(async () => {
    const source = join(root, "fake.ts");
    writeFileSync(
        source,
        `import {appendFileSync,existsSync,readFileSync} from 'node:fs';
import {join,dirname} from 'node:path';
const root=dirname(process.execPath);const args=process.argv.slice(2);
appendFileSync(join(root,'commands.jsonl'),JSON.stringify(args)+'\\n');
if(args[1]==='list'&&existsSync(join(root,'fail-'+args[0]))){console.error('recorded '+args[0]+' list failure');process.exit(1);}
if(args[0]==='auth'&&args[1]==='whoami'&&readFileSync(join(root,'login'),'utf8')==='missing')process.exit(1);
if(args[0]==='apps'&&args[1]==='list')console.log(JSON.stringify([{Name:'occupied'}]));
else if(args[0]==='machine'&&args[1]==='list')console.log(readFileSync(join(root,'machines.json'),'utf8'));
else if(args[0]==='ips'&&args[1]==='list')console.log(readFileSync(join(root,'ips.json'),'utf8'));
else if(args[0]==='volumes'&&args[1]==='list')console.log(readFileSync(join(root,'volumes.json'),'utf8'));
else if(args[0]==='orgs')console.log('{"personal":"Personal"}');
else if(args[0]==='deploy'&&existsSync(join(root,'fail-deploy'))){console.error('recorded deployment failed');process.exit(1);}`,
    );
    writeFileSync(join(root, "server.ts"), "export const state = 1;");
    writeFileSync(join(root, "login"), "present");
    writeFileSync(join(root, "machines.json"), JSON.stringify(machines));
    writeFileSync(join(root, "ips.json"), JSON.stringify(ips));
    writeFileSync(join(root, "volumes.json"), JSON.stringify([volume]));
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
            "volumes list",
            "machine list",
            "ips list",
            "deploy --app",
            "volumes list",
            "machine list",
        ]);
    } finally {
        await removeTestDirectory(work);
        if (token === undefined) delete process.env.FLY_API_TOKEN;
        else process.env.FLY_API_TOKEN = token;
    }
});

for (const [label, inventory, addresses, reason, inspect] of [
    ["zero Machines", [], ips, "Machine count 0", "machine list"],
    [
        "dedicated CPU",
        [{ ...machines[0], config: { guest: { ...guest, ["cpu_kind"]: "performance" } } }],
        ips,
        "Machine size",
        "machine list",
    ],
    [
        "extra CPU",
        [{ ...machines[0], config: { guest: { ...guest, cpus: 2 } } }],
        ips,
        "Machine size",
        "machine list",
    ],
    [
        "extra memory",
        [{ ...machines[0], config: { guest: { ...guest, ["memory_mb"]: 512 } } }],
        ips,
        "Machine size",
        "machine list",
    ],
    [
        "GPU",
        [{ ...machines[0], config: { guest: { ...guest, gpus: 1 } } }],
        ips,
        "Machine size",
        "machine list",
    ],
    [
        "GPU kind",
        [{ ...machines[0], config: { guest: { ...guest, ["gpu_kind"]: "a100" } } }],
        ips,
        "Machine size",
        "machine list",
    ],
    [
        "missing Machine config",
        [{ id: volume.attached_machine_id }],
        ips,
        "Machine size",
        "machine list",
    ],
    [
        "unmanaged Machine",
        [{ ...machines[0], config: { ...machines[0]!.config, metadata: {} } }],
        ips,
        "not managed by Fly Launch",
        "machine list",
    ],
    [
        "different process group",
        [
            {
                ...machines[0],
                config: {
                    ...machines[0]!.config,
                    metadata: { ["fly_platform_version"]: "v2", ["fly_process_group"]: "other" },
                },
            },
        ],
        ips,
        "not managed by Fly Launch",
        "machine list",
    ],
    [
        "legacy release command Machine",
        [
            {
                ...machines[0],
                config: {
                    ...machines[0]!.config,
                    metadata: {
                        ...machines[0]!.config.metadata,
                        ["process_group"]: "release_command",
                    },
                },
            },
        ],
        ips,
        "not managed by Fly Launch",
        "machine list",
    ],
    ["invalid Machine inventory", {}, ips, "expected an inventory array", "machine list"],
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
                        (row) =>
                            ["apps", "machine", "ips", "volumes"].includes(row[0]) &&
                            row[1] === "list",
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

test("an existing app with missing, extra, or mismatched storage is not redeployed", async () => {
    const work = mkdtempSync(join(tmpdir(), "shallot-fly-volume-refuse-"));
    try {
        for (const volumes of [
            [],
            [{ ...volume, ["size_gb"]: 2 }],
            [{ ...volume, name: "other" }],
            [{ ...volume, region: "ord" }],
            [{ ...volume, ["attached_machine_id"]: null }],
            [{ ...volume, id: "vol_2n0l9vl2l9pr635a" }],
        ]) {
            writeFileSync(join(root, "volumes.json"), JSON.stringify(volumes));
            writeFileSync(commands, "");
            await expect(
                deployFlyServer(
                    root,
                    work,
                    { target: "fly", app: "occupied" },
                    false,
                    true,
                    ask,
                    cli,
                ),
            ).rejects.toThrow("expected one 1 GB shallot_data volume in iad mounted at /data");
            expect(existsSync(join(work, "fly.toml"))).toBe(false);
            expect(readFileSync(commands, "utf8")).not.toContain('"deploy"');
        }
    } finally {
        writeFileSync(join(root, "volumes.json"), JSON.stringify([volume]));
        await removeTestDirectory(work);
    }
});

test.each([false, true])(
    "preflight reports excess volumes and any excess Machines (Machines: %s)",
    async (extraMachine) => {
        const work = mkdtempSync(join(tmpdir(), "shallot-fly-preflight-"));
        writeFileSync(commands, "");
        writeFileSync(
            join(root, "volumes.json"),
            JSON.stringify([volume, { ...volume, id: "vol_2n0l9vl2l9pr635f" }]),
        );
        if (extraMachine)
            writeFileSync(
                join(root, "machines.json"),
                JSON.stringify([...machines, { ...machines[0], id: "7591857c12836e" }]),
            );
        try {
            const deployed = deployFlyServer(
                root,
                work,
                { target: "fly", app: "occupied" },
                false,
                true,
                ask,
                cli,
            );
            await expect(deployed).rejects.toThrow("vol_2n0l9vl2l9pr635f: $0.15/month");
            await expect(deployed).rejects.toThrow(
                "volumes destroy vol_2n0l9vl2l9pr635f --app occupied",
            );
            if (extraMachine) {
                await expect(deployed).rejects.toThrow("7591857c12836e: about $2.19/month");
                await expect(deployed).rejects.toThrow(
                    "machine destroy 7591857c12836e --app occupied",
                );
            }
            expect(existsSync(join(work, "fly.toml"))).toBe(false);
            expect(readFileSync(commands, "utf8")).not.toMatch(/"(?:deploy|destroy)"/);
        } finally {
            writeFileSync(join(root, "volumes.json"), JSON.stringify([volume]));
            writeFileSync(join(root, "machines.json"), JSON.stringify(machines));
            await removeTestDirectory(work);
        }
    },
);

test.each([
    [true, true],
    [true, false],
    [false, true],
])(
    "successful deploy refuses missing resources (volume: %s, Machine: %s)",
    async (missingVolume, missingMachine) => {
        const work = mkdtempSync(join(tmpdir(), "shallot-fly-post-deploy-"));
        if (missingVolume) writeFileSync(join(root, "volumes.json"), "[]");
        if (missingMachine) writeFileSync(join(root, "machines.json"), "[]");
        try {
            const deployed = deployFlyServer(
                root,
                work,
                { target: "fly", app: "scratch-app" },
                false,
                false,
                ask,
                cli,
            );
            if (missingVolume) await expect(deployed).rejects.toThrow("missing volume");
            if (missingMachine) await expect(deployed).rejects.toThrow("missing Machine");
        } finally {
            writeFileSync(join(root, "volumes.json"), JSON.stringify([volume]));
            writeFileSync(join(root, "machines.json"), JSON.stringify(machines));
            await removeTestDirectory(work);
        }
    },
);

test("an attached volume with scheduled snapshots enabled refuses before deploy", async () => {
    const work = mkdtempSync(join(tmpdir(), "shallot-fly-snapshots-"));
    writeFileSync(
        join(root, "volumes.json"),
        JSON.stringify([{ ...volume, ["auto_backup_enabled"]: true }]),
    );
    try {
        await expect(
            deployFlyServer(root, work, { target: "fly", app: "occupied" }, false, true, ask, cli),
        ).rejects.toThrow("scheduled snapshots must be off");
        expect(existsSync(join(work, "fly.toml"))).toBe(false);
    } finally {
        writeFileSync(join(root, "volumes.json"), JSON.stringify([volume]));
        await removeTestDirectory(work);
    }
});

test.each([false, true])(
    "a Fly deploy that leaves two volumes reports both, bill and deletion commands (deploy failure: %s)",
    async (failed) => {
        const work = mkdtempSync(join(tmpdir(), "shallot-fly-post-deploy-"));
        writeFileSync(commands, "");
        const failFlag = join(root, "fail-deploy");
        if (failed) writeFileSync(failFlag, "");
        writeFileSync(
            join(root, "volumes.json"),
            JSON.stringify([volume, { ...volume, id: "vol_2n0l9vl2l9pr635f" }]),
        );
        try {
            const deployed = deployFlyServer(
                root,
                work,
                { target: "fly", app: "scratch-app" },
                false,
                false,
                ask,
                cli,
            );
            await expect(deployed).rejects.toThrow("2 volumes");
            await expect(deployed).rejects.toThrow("vol_2n0l9vl2l9pr635d: $0.15/month");
            await expect(deployed).rejects.toThrow("vol_2n0l9vl2l9pr635f: $0.15/month");
            await expect(deployed).rejects.toThrow(
                "volumes destroy vol_2n0l9vl2l9pr635d --app scratch-app",
            );
            await expect(deployed).rejects.toThrow(
                "volumes destroy vol_2n0l9vl2l9pr635f --app scratch-app",
            );
            if (failed) await expect(deployed).rejects.toThrow("recorded deployment failed");
            const calls = readFileSync(commands, "utf8")
                .trim()
                .split("\n")
                .map((line) => JSON.parse(line));
            expect(calls.slice(-3).map((args) => args[0])).toEqual([
                "deploy",
                "volumes",
                "machine",
            ]);
            expect(calls.some((args) => args.includes("destroy"))).toBe(false);
        } finally {
            rmSync(failFlag, { force: true });
            writeFileSync(join(root, "volumes.json"), JSON.stringify([volume]));
            await removeTestDirectory(work);
        }
    },
);

test.each([
    ["machine", "vol_2n0l9vl2l9pr635d", "vol_2n0l9vl2l9pr635f"],
    ["volumes", "7591857c12836d", "7591857c12836e"],
])(
    "a failed %s list still reports both resources from the other list",
    async (failed, first, second) => {
        const work = mkdtempSync(join(tmpdir(), "shallot-fly-post-deploy-"));
        const failFlag = join(root, `fail-${failed}`);
        writeFileSync(failFlag, "");
        writeFileSync(
            join(root, "volumes.json"),
            JSON.stringify([volume, { ...volume, id: "vol_2n0l9vl2l9pr635f" }]),
        );
        writeFileSync(
            join(root, "machines.json"),
            JSON.stringify([...machines, { ...machines[0], id: "7591857c12836e" }]),
        );
        try {
            const deployed = deployFlyServer(
                root,
                work,
                { target: "fly", app: "scratch-app" },
                false,
                false,
                ask,
                cli,
            );
            await expect(deployed).rejects.toThrow(`recorded ${failed} list failure`);
            await expect(deployed).rejects.toThrow(`${first}:`);
            await expect(deployed).rejects.toThrow(`${second}:`);
        } finally {
            rmSync(failFlag, { force: true });
            writeFileSync(join(root, "volumes.json"), JSON.stringify([volume]));
            writeFileSync(join(root, "machines.json"), JSON.stringify(machines));
            await removeTestDirectory(work);
        }
    },
);

test.each([
    ["machine", [null]],
    ["volumes", [volume, { ...volume, id: "vol_2n0l9vl2l9pr635e", ["size_gb"]: -1 }]],
    ["volumes", [volume, { ...volume, id: "arbitrary-token" }]],
    ["machine", [...machines, { ...machines[0], id: "arbitrary-token" }]],
] as const)(
    "malformed %s rows make the count unknown without hiding valid rows (%j)",
    async (kind, rows) => {
        const work = mkdtempSync(join(tmpdir(), "shallot-fly-post-deploy-"));
        const path = join(root, kind === "machine" ? "machines.json" : "volumes.json");
        const original = readFileSync(path, "utf8");
        writeFileSync(path, JSON.stringify(rows));
        try {
            const deployed = deployFlyServer(
                root,
                work,
                { target: "fly", app: "scratch-app" },
                false,
                false,
                ask,
                cli,
            );
            const error = await deployed.then(
                () => "",
                (error: unknown) => String(error),
            );
            expect(error).toContain(`${kind === "machine" ? "Machine" : "volume"} count unknown`);
            expect(error).toContain(`${volume.id}: $0.15/month`);
            if (rows[0] !== null) expect(error).toContain(`${machines[0]!.id}: about $2.19/month`);
            expect(error).not.toContain("$-0.15");
            expect(error).not.toContain("destroy arbitrary-token");
        } finally {
            writeFileSync(path, original);
            await removeTestDirectory(work);
        }
    },
);

test("a Fly deploy that leaves two Machines names both and the always-on cost", async () => {
    const work = mkdtempSync(join(tmpdir(), "shallot-fly-post-deploy-"));
    writeFileSync(
        join(root, "machines.json"),
        JSON.stringify([...machines, { ...machines[0], id: "7591857c12836e" }]),
    );
    try {
        const deployed = deployFlyServer(
            root,
            work,
            { target: "fly", app: "scratch-app" },
            false,
            false,
            ask,
            cli,
        );
        await expect(deployed).rejects.toThrow("2 Machines");
        await expect(deployed).rejects.toThrow("7591857c12836d: about $2.19/month");
        await expect(deployed).rejects.toThrow("7591857c12836e: about $2.19/month");
        await expect(deployed).rejects.toThrow("machine destroy 7591857c12836d --app scratch-app");
        await expect(deployed).rejects.toThrow("machine destroy 7591857c12836e --app scratch-app");
    } finally {
        writeFileSync(join(root, "machines.json"), JSON.stringify(machines));
        await removeTestDirectory(work);
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

test("missing Fly login in a terminal runs login before app checks", async () => {
    const token = process.env.FLY_API_TOKEN;
    delete process.env.FLY_API_TOKEN;
    writeFileSync(join(root, "login"), "missing");
    writeFileSync(commands, "");
    try {
        await expect(
            deployFlyServer(root, root, { target: "fly", app: "occupied" }, true, false, ask, cli),
        ).rejects.toThrow("already exists");
        const calls = readFileSync(commands, "utf8")
            .trim()
            .split("\n")
            .map((line) => JSON.parse(line));
        expect(calls).toEqual([
            ["auth", "whoami"],
            ["auth", "login"],
            ["apps", "list", "--json"],
        ]);
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
