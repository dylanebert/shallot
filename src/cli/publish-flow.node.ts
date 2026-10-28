import { afterAll, beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import {
    linkSync,
    mkdirSync,
    mkdtempSync,
    readdirSync,
    readFileSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join, resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { chromium } from "playwright";
import { bunChild } from "../../scripts/bun-child";
import { removeTestDirectory } from "../../scripts/test-directory";
import { CROSS_ORIGIN_ISOLATION } from "../project/isolation.js";
import { WRANGLER_VERSION } from "./publish-cloudflare";

setDefaultTimeout(TIMEOUT.node);
const repo = resolve(import.meta.dir, "../..");
const accountA = "a".repeat(32),
    accountB = "b".repeat(32);

const RECORD = `const record=(entry)=>fs.appendFileSync(path.join(root,'commands.jsonl'),JSON.stringify(entry)+'\\n');`;
const tools = mkdtempSync(join(tmpdir(), "shallot-publish-tools-"));
const suffix = process.platform === "win32" ? ".exe" : "";
/**
 * The fixture's wrangler and bun: one native executable, linked into each fixture's bin, that reads the
 * fixture's root as its bin's parent. Native tools keep PATH interception independent of POSIX shebangs.
 * Source-derived Wrangler 4.151.3 answers: whoami JSON, versions/list JSON, command-failed code 10007,
 * and deploy JSONL. No credentials or provider API.
 */
let tool = "";
beforeAll(async () => {
    const source = join(tools, "tool.ts");
    writeFileSync(
        source,
        `const fs=require('node:fs'),path=require('node:path'); const root=path.dirname(path.dirname(process.execPath)); ${RECORD}
const args=process.argv.slice(2);
if(path.basename(process.execPath).replace(/\\.exe$/, '')==='bun') {
if(JSON.stringify(args)!==JSON.stringify(['add','--dev','wrangler@${WRANGLER_VERSION}']))throw Error('Unexpected install');
record({command:'install'});const p=path.join(process.cwd(),'package.json'),m=JSON.parse(fs.readFileSync(p));m.devDependencies={wrangler:'${WRANGLER_VERSION}'};fs.writeFileSync(p,JSON.stringify(m,null,4));fs.cpSync(path.join(root,'wrangler'),path.join(process.cwd(),'node_modules/wrangler'),{recursive:true});process.exit(0);
}
const state=JSON.parse(fs.readFileSync(path.join(root,'answers.json')));
const config=JSON.parse(fs.readFileSync(args[args.indexOf('--config')+1]));
record({command:args[0],config,linked:JSON.parse(fs.readFileSync(path.join(process.cwd(),'package.json'))).shallot.publish});
const replies=JSON.parse(fs.readFileSync(${JSON.stringify(join(import.meta.dir, "fixtures/publish-wrangler.json"))}));
const replay=(reply)=>{for(const entry of reply.entries??[])fs.appendFileSync(process.env.WRANGLER_OUTPUT_FILE_PATH,JSON.stringify(entry)+'\\n');process.stdout.write(reply.stdout??'');process.stderr.write(reply.stderr??'');process.exit(reply.status);};
if(args[0]==='whoami')replay(state.account==='${accountA}'?replies.accountA:replies.accountB);
if(args[0]==='versions')replay(state.collision?replies.exists:replies.absent);
if(args[0]==='deploy'){if(!fs.existsSync(path.join(config.assets.directory,'_headers')))throw Error('missing headers');fs.writeFileSync(path.join(root,'deployed.json'),JSON.stringify(config));replay(state.fail?replies.failed:replies.deployed);}
throw Error('Unexpected provider command');`,
    );
    const executable = join(tools, `tool${suffix}`);
    const compiled = await Bun.build({ entrypoints: [source], compile: { outfile: executable } });
    if (!compiled.success) throw new Error(compiled.logs.map(String).join("\n"));
    tool = executable;
});
afterAll(() => removeTestDirectory(tools));

async function fixture() {
    const root = mkdtempSync(join(tmpdir(), "shallot-publish-flow-"));
    const project = join(root, "game"),
        bin = join(root, "bin");
    for (const path of [
        project,
        bin,
        join(project, "node_modules/vite"),
        join(project, "node_modules/playwright"),
        join(root, "wrangler"),
    ])
        mkdirSync(path, { recursive: true });
    const manifest = {
        name: "fixture-game",
        scripts: { build: "vite build" },
        shallot: { other: 42 },
    };
    const packagePath = join(project, "package.json");
    writeFileSync(packagePath, JSON.stringify(manifest, null, 4));
    writeFileSync(join(project, "vite.config.js"), "export default {};\n");
    writeFileSync(join(project, "untouched.txt"), "creator's file\n");
    const statePath = join(root, "answers.json"),
        records = join(root, "commands.jsonl");
    const state = { account: accountA, fail: false, collision: false };
    const save = () => writeFileSync(statePath, JSON.stringify(state));
    save();
    writeFileSync(records, "");
    const record = `const fs=require('node:fs'),path=require('node:path'); const root=${JSON.stringify(root)}; ${RECORD}`;
    writeFileSync(
        join(root, "wrangler/package.json"),
        JSON.stringify({
            name: "wrangler",
            version: WRANGLER_VERSION,
            bin: { wrangler: "cli.cjs" },
        }),
    );
    writeFileSync(
        join(root, "wrangler/cli.cjs"),
        `const r=require('node:child_process').spawnSync('wrangler',process.argv.slice(2),{stdio:'inherit',env:process.env});process.exit(r.status??1);`,
    );
    for (const name of ["wrangler", "bun"]) linkSync(tool, join(bin, `${name}${suffix}`));
    writeFileSync(
        join(project, "node_modules/vite/package.json"),
        JSON.stringify({ name: "vite", bin: { vite: "cli.cjs" } }),
    );
    writeFileSync(
        join(project, "node_modules/vite/cli.cjs"),
        `${record}
record({command:'build'});const directory=process.argv[process.argv.indexOf('--outDir')+1];fs.mkdirSync(directory,{recursive:true});fs.writeFileSync(path.join(directory,'index.html'),'<link rel="icon" href="data:,"><script>dispatchEvent(new CustomEvent("shallot:startup",{detail:{kind:"app"}}))</script>');`,
    );
    const server = Bun.serve({
        hostname: "127.0.0.1",
        port: 0,
        fetch(request) {
            const config = JSON.parse(readFileSync(join(root, "deployed.json"), "utf8"));
            const path = new URL(request.url).pathname;
            if (path !== "/" && path !== "/index.html")
                return new Response("missing", { status: 404 });
            return new Response(readFileSync(join(config.assets.directory, "index.html")), {
                headers: { ...CROSS_ORIGIN_ISOLATION, "Content-Type": "text/html" },
            });
        },
    });
    // Substitute only the fixture provider origin's transport. Both API requests
    // and page navigation still use real Playwright against the local boundary.
    writeFileSync(
        join(project, "node_modules/playwright/package.json"),
        JSON.stringify({ name: "playwright", type: "module", main: "index.js" }),
    );
    writeFileSync(
        join(project, "node_modules/playwright/index.js"),
        `import {publishTestChromium} from ${JSON.stringify(join(repo, "scripts/publish-browser.ts"))};
export const chromium={async launch(){const browser=await publishTestChromium(),create=browser.newContext.bind(browser);const map=(url)=>{const u=new URL(url);if(u.origin!=='https://fixture.workers.dev')throw Error('Unexpected external request');return ${JSON.stringify(server.url.origin)}+u.pathname;};browser.newContext=async(...args)=>{const context=await create(...args),get=context.request.get.bind(context.request),page=context.newPage.bind(context);context.request.get=(url,...args)=>get(map(url),...args);context.newPage=async()=>{const p=await page(),go=p.goto.bind(p);p.goto=(url,...args)=>go(map(url),...args);return p;};return context;};return browser;}};`,
    );
    const runner = join(root, "run.ts");
    writeFileSync(
        runner,
        `import {publishProject} from ${JSON.stringify(join(repo, "src/cli/publish.ts"))};try{await publishProject(process.argv[2]);}catch(error){console.error(error.message);process.exitCode=1;}`,
    );
    const run = async (account?: string) => {
        const env = Object.fromEntries(
            Object.entries(process.env).filter(([key]) => !/^(CLOUDFLARE_|CF_)/.test(key)),
        );
        const child = Bun.spawn({
            ...bunChild([runner, project]),
            cwd: project,
            env: {
                ...env,
                PATH: `${bin}${delimiter}${process.env.PATH}`,
                CI: "true",
                SHALLOT_TEST_CHROMIUM: chromium.executablePath(),
                HOME: join(root, "home"),
                XDG_CONFIG_HOME: join(root, "home"),
                ...(account ? { CLOUDFLARE_ACCOUNT_ID: account } : {}),
            },
            stdin: "ignore",
            stdout: "pipe",
            stderr: "pipe",
        });
        const [code, stdout, stderr] = await Promise.all([
            child.exited,
            new Response(child.stdout).text(),
            new Response(child.stderr).text(),
        ]);
        return { code, stdout, stderr };
    };
    const commands = () =>
        readFileSync(records, "utf8")
            .trim()
            .split("\n")
            .filter(Boolean)
            .map((line) => JSON.parse(line));
    const snapshot = () =>
        Object.fromEntries(
            readdirSync(project)
                .filter((name) => name !== "node_modules")
                .map((name) => [name, readFileSync(join(project, name), "utf8")]),
        );
    return {
        project,
        manifest,
        packagePath,
        state,
        save,
        run,
        commands,
        snapshot,
        async close() {
            await server.stop(true);
            await removeTestDirectory(root);
        },
    };
}

test("publishProject changes only the link and installed devDependency; a saved account persists unless the environment conflicts", async () => {
    const f = await fixture();
    try {
        const before = f.snapshot();
        const result = await f.run();
        expect(result.code, result.stderr).toBe(0);
        expect(result.stdout.trim()).toBe("https://fixture.workers.dev");
        const link = { target: "cloudflare", name: "fixture-game", accountId: accountA };
        expect(JSON.parse(readFileSync(f.packagePath, "utf8"))).toEqual({
            ...f.manifest,
            devDependencies: { wrangler: WRANGLER_VERSION },
            shallot: { other: 42, publish: link },
        });
        const after = f.snapshot();
        expect(Object.keys(after)).toEqual(Object.keys(before));
        for (const name of Object.keys(before))
            if (name !== "package.json") expect(after[name]).toBe(before[name]);
        expect(f.commands().map((entry) => entry.command)).toEqual([
            "install",
            "whoami",
            "build",
            "versions",
            "deploy",
        ]);
        const deployment = f.commands().find((entry) => entry.command === "deploy");
        expect(deployment.linked).toBeUndefined();
        expect(deployment.config.account_id).toBe(accountA);
        // A saved account stays explicit even if whoami lists a different sole account.
        f.state.account = accountB;
        f.save();
        const switched = await f.run();
        expect(switched.code, switched.stderr).toBe(0);
        expect(
            f
                .commands()
                .filter((entry) => entry.command === "deploy")
                .at(-1).config.account_id,
        ).toBe(accountA);
        f.state.account = accountA;
        f.save();
        const explicit = await f.run(accountB);
        expect(explicit.code).toBe(1);
        expect(explicit.stderr).toContain(accountA);
        expect(explicit.stderr).toContain(accountB);
        expect(f.commands().filter((entry) => entry.command === "deploy")).toHaveLength(2);
        expect(f.snapshot()).toEqual(after);
    } finally {
        await f.close();
    }
});

test("a failed deploy creates no ownership; retry rebuilds and checks absence again", async () => {
    const f = await fixture();
    try {
        f.state.fail = true;
        f.save();
        const failed = await f.run();
        expect(failed.code).toBe(1);
        expect(failed.stderr).toContain("recorded deploy failure");
        expect(JSON.parse(readFileSync(f.packagePath, "utf8")).shallot.publish).toBeUndefined();
        f.state.collision = true;
        f.save();
        const retry = await f.run();
        expect(retry.code).toBe(1);
        expect(retry.stderr).toContain('Worker "fixture-game" already exists');
        expect(JSON.parse(readFileSync(f.packagePath, "utf8")).shallot.publish).toBeUndefined();
        expect(f.commands().map((entry) => entry.command)).toEqual([
            "install",
            "whoami",
            "build",
            "versions",
            "deploy",
            "whoami",
            "build",
            "versions",
        ]);
    } finally {
        await f.close();
    }
});
