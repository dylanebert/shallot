import { expect, setDefaultTimeout, test } from "bun:test";
import {
    chmodSync,
    mkdirSync,
    mkdtempSync,
    readFileSync,
    realpathSync,
    rmSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { basename, delimiter, resolve } from "node:path";
import { CEILING } from "@dylanebert/shallot/testing";
import { BUN_CHILD, bunCommand } from "./bun-child";
import { packedInstall } from "./packed-install-spawn";

setDefaultTimeout(CEILING.node);
const repo = resolve(import.meta.dir, "..");

test("installed packed CLI preserves editable favicon and native PNG at a nested web base", () => {
    const root = realpathSync.native(mkdtempSync(resolve(tmpdir(), "shallot-icons-packed-")));
    const installer = resolve(root, "cli");
    const project = resolve(root, "game");
    const env = { ...process.env };
    const run = (args: string[], cwd = project) => {
        const result = Bun.spawnSync(bunCommand(args), {
            ...BUN_CHILD,
            cwd,
            env,
            stdout: "pipe",
            stderr: "pipe",
        });
        expect(result.exitCode, result.stderr.toString()).toBe(0);
        return result.stdout.toString();
    };
    try {
        const archive = run(
            ["pm", "pack", "--ignore-scripts", "--quiet", "--destination", root],
            repo,
        ).trim();
        const spec = `file:../${basename(archive)}`;
        mkdirSync(installer);
        writeFileSync(
            resolve(installer, "package.json"),
            JSON.stringify({
                name: "packed-icon-cli",
                private: true,
                dependencies: { "@dylanebert/shallot": spec },
            }),
        );
        packedInstall(
            ["install", "--linker=isolated", "--no-progress"],
            installer,
            "icon CLI install",
        );
        const installed = resolve(installer, "node_modules/@dylanebert/shallot");
        const manifest = JSON.parse(readFileSync(resolve(installed, "package.json"), "utf8"));
        const cli = resolve(installed, manifest.bin.shallot);
        run([cli, "create", project], installer);
        const pkgPath = resolve(project, "package.json");
        const pkg = JSON.parse(readFileSync(pkgPath, "utf8"));
        pkg.dependencies["@dylanebert/shallot"] = spec;
        writeFileSync(pkgPath, JSON.stringify(pkg));
        packedInstall(
            ["install", "--linker=isolated", "--no-progress"],
            project,
            "icon consumer install",
        );
        const projectCli = resolve(
            project,
            "node_modules/@dylanebert/shallot",
            manifest.bin.shallot,
        );
        const tools = resolve(root, "tools");
        const runtime = resolve(root, "cef");
        mkdirSync(tools);
        mkdirSync(runtime);
        writeFileSync(resolve(runtime, "libcef.so"), "fixture CEF library, never launched");
        const cargoSource = `const fs=require('node:fs'),path=require('node:path');
const args=process.argv.slice(2),target=args[args.indexOf('--target')+1];
const out=path.join(process.env.CARGO_TARGET_DIR,target,'debug');fs.mkdirSync(out,{recursive:true});
fs.writeFileSync(path.join(out,'shallot-native'),'fixture shell, never launched');
console.log(JSON.stringify({reason:'build-script-executed',package_id:'path+file:///fixture#shallot-native@0.10.0',env:[['SHALLOT_CEF_DIR',process.env.FIXTURE_RUNTIME]]}));`;
        if (process.platform === "win32") {
            writeFileSync(resolve(tools, "cargo.js"), cargoSource);
            writeFileSync(
                resolve(tools, "cargo.cmd"),
                `@echo off\r\n"${process.execPath}" "%~dp0cargo.js" %*\r\n`,
            );
        } else {
            writeFileSync(resolve(tools, "cargo"), `#!${process.execPath}\n${cargoSource}`);
            chmodSync(resolve(tools, "cargo"), 0o755);
        }
        env.PATH = `${tools}${delimiter}${env.PATH}`;
        env.FIXTURE_RUNTIME = runtime;
        env.CEF_PATH = undefined;
        env.WSL_INTEROP = undefined;
        const standardSvg = readFileSync(resolve(installed, "examples/minimal/public/icon.svg"));
        const standardPng = readFileSync(resolve(installed, "assets/icon-1024.png"));
        expect(readFileSync(resolve(project, "public/icon.svg"))).toEqual(standardSvg);
        run([projectCli, "build", "--target", "linux", "--portable", "--base", "/nested/"]);
        const out = resolve(project, "build/linux/debug-portable");
        expect(readFileSync(resolve(project, "dist/index.html"), "utf8")).toContain(
            'href="/nested/icon.svg"',
        );
        expect(readFileSync(resolve(out, "dist/icon.svg"))).toEqual(standardSvg);
        expect(readFileSync(resolve(out, "dist/icon.png"))).toEqual(standardPng);
        const customSvg = Buffer.from(
            '<svg xmlns="http://www.w3.org/2000/svg"><circle r="4" fill="red"/></svg>',
        );
        const customPng = Buffer.from(standardPng);
        // The trailing ancillary bytes distinguish consumer ownership without changing the PNG image.
        const customBytes = Buffer.concat([customPng, Buffer.from("consumer PNG")]);
        writeFileSync(resolve(project, "public/icon.svg"), customSvg);
        writeFileSync(resolve(project, "public/icon.png"), customBytes);
        run([projectCli, "build", "--target", "linux", "--portable", "--base", "/nested/"]);
        expect(readFileSync(resolve(out, "dist/icon.svg"))).toEqual(customSvg);
        expect(readFileSync(resolve(out, "dist/icon.png"))).toEqual(customBytes);
        expect(readFileSync(resolve(project, "dist/index.html"), "utf8")).toContain(
            'href="/nested/icon.svg"',
        );
        const htmlPath = resolve(project, "index.html");
        writeFileSync(
            htmlPath,
            readFileSync(htmlPath, "utf8").replace(/<link rel="icon"[^>]+>/, ""),
        );
        run(["run", "build", "--base", "/nested/"]);
        expect(readFileSync(resolve(project, "dist/index.html"), "utf8")).not.toContain(
            'rel="icon"',
        );
        console.info(
            "packed consumer: installed tarball, public CLI create/build, nested favicon default/custom, no injected favicon, native PNG default/custom; fake Cargo, no launch",
        );
    } finally {
        rmSync(root, { recursive: true, force: true });
    }
});
