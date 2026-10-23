import { readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";
import { nativeHostDiagnostic } from "../native";
import { resolveViteCli, startViteDev } from "../project";
import { buildProject, type NativeTarget } from "./build";
import { previewProject } from "./preview";

const desktopStatus = `
  Desktop status (experimental)
    linux x86_64   Use --portable; WebKitGTK does not enable WebGPU by default.
    mac arm64      Without --portable, needs macOS 26 for WebGPU.
    windows x64    --portable is checked only on a software GPU; without it, unchecked.
`;

const usage = `
  shallot — tools for a Shallot project

  Shallot is a library: Vite runs your project and Bun tests it.

  Your loop
    bun run dev                  run the Vite project
    bun test                     run cheap and GPU tests
    bun test ./<file>            run a Node or oracle file by path
    bunx playwright test         run browser tests

  Commands
    create <name>                create a project from minimal or an example
    migrate                      print the 0.10 to 0.10.1 migration guide
    publish [--target <name>]    build, publish and verify a public web game
    dev --target <platform>      develop a desktop app
    build --target <platform>    build a desktop app
    preview --target <platform>  launch a desktop build

  Desktop options
    --target <platform>          windows, mac, linux
    --release                    optimized native build
    --portable                   bundle the Chromium runtime (CEF)
    Desktop targets are experimental; see \`shallot build --help\` for status.

  Help
    -V, --version                print the installed version
    --skill                      print the Shallot project skill
    shallot <command> --help
    shallot help [command]
`;

const commandUsage = {
    publish: `
  shallot publish [--target cloudflare|itch]

  Build with the project's Vite config, preflight the target's limits, publish,
  wait until the upload is live and verify the public link anonymously with
  Playwright. stdout is the stable play link; progress and verification go to
  stderr. The target is saved in package.json's shallot.publish.

  cloudflare (the default): Workers Static Assets, 20,000 files and 25 MiB per file.
  Wrangler requires Node.js 22 or newer on PATH.
  Missing pinned Wrangler is added to devDependencies inline. After a successful
  deploy, shallot.publish (target, name, accountId) is saved in package.json.
  Wrangler owns login and credentials. The account must match the link.
  First links are checked for name collisions after building, immediately before
  deployment.
  On a collision a terminal can choose another name; without a TTY, in CI or for
  an agent, publish refuses. A hand-written link deliberately authorizes an update.
  Non-interactive login: CLOUDFLARE_API_TOKEN; CLOUDFLARE_ACCOUNT_ID when ambiguous.
  Least CI token: custom, Account → Workers Scripts → Edit on the one account.
  The "Edit Cloudflare Workers" template is broader.
  Provider commands run from the project with its pinned Wrangler. Substitute
  <account> and <name> from shallot.publish.accountId and .name.
  These commands assume no Wrangler config in the project or any directory above
  it, because its account_id overrides the link's account environment variable.
  Login: CLOUDFLARE_ACCOUNT_ID=<account> bunx wrangler login
  Remote login: CLOUDFLARE_ACCOUNT_ID=<account> bunx wrangler login --device
  Preview: CLOUDFLARE_ACCOUNT_ID=<account> bunx wrangler dev --assets <web-build-directory>
  "Roll back": CLOUDFLARE_ACCOUNT_ID=<account> bunx wrangler rollback --name <name>
  "Take the game down": CLOUDFLARE_ACCOUNT_ID=<account> bunx wrangler delete --name <name>
  Confirm with the person before deleting: Wrangler auto-confirms without a TTY
  or in CI. Deletion cannot be undone.
  "See what is published": https://dash.cloudflare.com/<account>/workers/overview
  lists every Worker in the account; Wrangler has no account-wide list command.

  itch: the page's html5 channel, 1,000 files, 200 MB per file, 500 MB in all.
  Create the page on itch.io with Kind of project HTML; shallot.publish (target
  itch, name <user>/<game>) names it. itch.io's butler is fetched at its pinned
  version into Shallot's cache and owns login and credentials. Verification plays
  the build's document on html.itch.zone, which the page's embed frames; itch.io
  serves a new build once its page is next viewed. Until then, interactive publish
  refuses; without interaction it succeeds but reports the build unverified,
  naming the page and build. Open the page, then publish again to verify it. On the
  page's Edit page, tick "This file will be played in the browser" and
  "SharedArrayBuffer support", and set Visibility to Public or Restricted, not
  Draft; verification cannot see these, so publish names them.
  Non-interactive login: butler's stored credentials or BUTLER_API_KEY.
  Use the quoted cached butler path printed by publish as <butler>, and
  shallot.publish.name as <user>/<game>:
  Login: "<butler>" login
  Preview locally: bun run preview
  "See what is published": "<butler>" status <user>/<game>
  All pages and Edit: https://itch.io/dashboard, choose Edit for <game>.
  "Take the game down": confirm with the person, then set Visibility to Draft
  on that Edit page. butler cannot unpublish or delete a page.
  "Roll back": check out an earlier commit, then bunx shallot publish --target itch
  Cleanup, preview, rollback and listing stay the providers'; Shallot adds no
  unpublish or list command.
  butler's login key has wharf scope: builds on every project the identity can
  reach, not one game. An account-settings API key can be broader still.

  Publish adds no credentials to the build. Vite's client values, public files
  and source maps can still disclose what the project puts there. Never put secrets in them.

  --target <name>       Choose and save the target: cloudflare or itch
  -h, --help            Show this help
`,
    dev: `
  shallot dev --target <platform> [dir] [Vite options]

  Develop with hot reload in a native desktop shell. Web projects run \`bun run dev\`.
  The directory defaults to the current directory (.). Other arguments go to Vite unchanged.

  Common examples
    bun run dev
    shallot dev --target mac
    shallot dev --target mac --portable

  Shallot options
    --target <platform>   windows, mac, linux
    --portable            Bundle the Chromium runtime (CEF)
    -h, --help            Show this help
${desktopStatus}`,
    build: `
  shallot build --target <platform> [dir] [Vite options]

  Build a native desktop app with the project's Vite config. Web projects run \`bun run build\`.
  The directory defaults to the current directory (.). Other arguments go to Vite unchanged.

  Common examples
    bun run build
    shallot build --target mac
    shallot build --target linux --portable

  Shallot options
    --target <platform>   windows, mac, linux
    --release             Optimized native build
    --portable            Bundle the Chromium runtime (CEF) instead of the system webview.
    -h, --help            Show this help

${desktopStatus}
  Native requirements
    Native release builds use a prebuilt shell from GitHub Releases when available. A miss (404,
    offline, checksum mismatch, or source checkout) falls back to compiling the Rust native host from
    source. Debug builds always compile from source. Source builds need the Rust toolchain
    (https://rustup.rs) and, per target:
      mac       macOS host with Xcode command line tools
      linux     Linux host; --portable only (WebKitGTK does not enable WebGPU by default), with libx11-dev
      windows   cargo-xwin to cross-compile; --portable needs a Windows host with Visual Studio,
                its C++ workload and ATL
    Portable builds download CEF on first build, or use CEF_PATH when set.
`,
    preview: `
  shallot preview --target <platform> [dir]

  Launch an existing native desktop build. Web projects run \`bun run preview\`.
  The directory defaults to the current directory (.).

  Common examples
    bun run preview
    shallot preview --target mac
    shallot preview --target linux --portable

  Shallot options
    --target <platform>   windows, mac, linux
    --release             Launch the optimized native build
    --portable            Launch the build with its bundled Chromium runtime (CEF)
    -h, --help            Show this help

${desktopStatus}
  Native targets use the requirements documented by 'shallot build --help'.
`,
    migrate: `
  shallot migrate

  Print the guide to migrating a 0.10 app to 0.10.1. It changes no files.

  Shallot options
    -h, --help   Show this help
`,
} as const;

export type CliArgs =
    | { kind: "create"; rest: string[] }
    | { kind: "publish"; target?: string }
    | { kind: "skill" }
    | { kind: "version" }
    | { kind: "migrate" }
    | { kind: "command-help"; command: keyof typeof commandUsage }
    | { kind: "unknown"; verb: string }
    | { kind: "usage"; exitCode: 0 | 1 }
    | {
          kind: "run";
          subcmd: "dev" | "build" | "preview";
          dir: string;
          target?: string;
          release: boolean;
          portable: boolean;
          viteArgs: string[];
      };

const PROJECT_VERBS = ["dev", "build", "preview"] as const;
const TARGETS: readonly NativeTarget[] = ["windows", "mac", "linux"];

function isNativeTarget(target: string): target is NativeTarget {
    return (TARGETS as readonly string[]).includes(target);
}

/** Separate Shallot's target flags; preserve every Vite argument and its order. */
export function parseCliArgs(raw: string[]): CliArgs {
    const separator = raw.indexOf("--");
    const options = separator < 0 ? raw : raw.slice(0, separator);
    if (options.some((arg) => arg === "--help" || arg === "-h")) {
        const command = raw[0] === "help" ? options[1] : options[0];
        if (command === "create") return { kind: "create", rest: ["--help"] };
        if (command && Object.hasOwn(commandUsage, command))
            return { kind: "command-help", command: command as keyof typeof commandUsage };
        return { kind: "usage", exitCode: 0 };
    }
    if (raw[0] === "help") {
        if (raw.length > 2) throw new Error(`unexpected argument: ${raw[2]}`);
        const command = raw[1];
        if (!command) return { kind: "usage", exitCode: 0 };
        if (command === "create") return { kind: "create", rest: ["--help"] };
        if (Object.hasOwn(commandUsage, command))
            return { kind: "command-help", command: command as keyof typeof commandUsage };
        return { kind: "unknown", verb: command };
    }
    const verb = raw[0];
    if (verb === "--version" || verb === "-V" || verb === "--skill") {
        if (raw.length > 1) throw new Error(`${verb} takes no arguments`);
        return { kind: verb === "--skill" ? "skill" : "version" };
    }
    if (verb === "create") return { kind: "create", rest: raw.slice(1) };
    if (verb === "publish") {
        if (raw.length === 1) return { kind: "publish" };
        if (raw.length === 3 && raw[1] === "--target" && raw[2] && !raw[2].startsWith("-"))
            return { kind: "publish", target: raw[2] };
        if (raw.length === 2 && raw[1].startsWith("--target=") && raw[1].slice(9))
            return { kind: "publish", target: raw[1].slice(9) };
        throw new Error("publish accepts only --target <name>");
    }
    if (verb === "migrate") {
        if (raw.length === 1) return { kind: "migrate" };
        throw new Error("migrate takes no arguments");
    }
    const subcmd = PROJECT_VERBS.find((command) => command === verb);
    if (!subcmd) {
        if (!verb) return { kind: "usage", exitCode: 0 };
        if (verb.startsWith("-")) throw new Error(`unknown option: ${verb}`);
        return { kind: "unknown", verb };
    }
    let target: string | undefined;
    let release = false;
    let portable = false;
    const viteArgs: string[] = [];

    for (let i = subcmd ? 1 : 0; i < raw.length; i++) {
        const arg = raw[i];
        if (arg === "--") {
            viteArgs.push(...raw.slice(i));
            break;
        }
        if (arg === "--target") {
            const value = raw[i + 1];
            if (!value || value.startsWith("-")) throw new Error("--target requires a platform");
            target = value;
            i++;
        } else if (arg.startsWith("--target=")) {
            target = arg.slice("--target=".length);
            if (!target) throw new Error("--target requires a platform");
        } else if (arg === "--release") {
            release = true;
        } else if (arg === "--portable") {
            portable = true;
        } else {
            viteArgs.push(arg);
        }
    }

    if (subcmd === "dev" && release) throw new Error("--release is not supported by dev");

    // A leading positional is Vite's root, expressed as the command's working directory.
    const dir = viteArgs[0] && !viteArgs[0].startsWith("-") ? viteArgs.shift()! : ".";
    if (subcmd === "preview" && viteArgs.length)
        throw new Error(`unexpected argument: ${viteArgs[0]}`);
    return { kind: "run", subcmd, dir, target, release, portable, viteArgs };
}

function helpHint(raw: string[]): string {
    const command =
        raw[0] && (raw[0] === "publish" || PROJECT_VERBS.some((verb) => verb === raw[0]))
            ? `shallot ${raw[0]}`
            : "shallot";
    return `See \`${command} --help\` for available options.`;
}

/** run the `shallot` CLI over argv (without the runtime and script) and exit with its status. */
export async function main(
    raw: string[],
    exit: (code: number) => never = process.exit,
): Promise<void> {
    let parsed: CliArgs;
    try {
        parsed = parseCliArgs(raw);
    } catch (e) {
        console.error(e instanceof Error ? e.message : String(e));
        console.error(helpHint(raw));
        exit(1);
    }

    if (parsed.kind === "usage") {
        console.log(usage);
        exit(parsed.exitCode);
    }
    if (parsed.kind === "command-help") {
        console.log(commandUsage[parsed.command]);
        exit(0);
    }
    if (parsed.kind === "version") {
        const { version } = JSON.parse(
            readFileSync(new URL("../../package.json", import.meta.url), "utf8"),
        );
        console.log(version);
        exit(0);
    }
    if (parsed.kind === "skill") {
        console.log(readFileSync(new URL("./skill.md", import.meta.url), "utf8").trimEnd());
        exit(0);
    }
    if (parsed.kind === "migrate") {
        console.log(readFileSync(new URL("../../MIGRATION.md", import.meta.url), "utf8").trimEnd());
        exit(0);
    }
    if (parsed.kind === "create") {
        const { runCreate } = await import("./create");
        exit(runCreate(parsed.rest));
    }
    if (parsed.kind === "unknown") {
        console.error(`unknown command: ${parsed.verb}`);
        console.error("See `shallot --help` for available commands.");
        exit(1);
    }

    if (parsed.kind === "publish") {
        try {
            const { publishProject } = await import("./publish");
            await publishProject(process.cwd(), parsed.target);
            exit(0);
        } catch (error) {
            console.error(error instanceof Error ? error.message : String(error));
            exit(1);
        }
    }

    const target = parsed.target;
    if (!target || target === "web") {
        console.error(
            `For web, run \`bun run ${parsed.subcmd}\`; for desktop, rerun with \`--target mac|windows|linux\`.`,
        );
        exit(1);
    }
    if (!isNativeTarget(target)) {
        console.error(`unknown target: ${target}`);
        console.error(`See \`shallot ${parsed.subcmd} --help\` for available targets and options.`);
        exit(1);
    }

    const hostDiagnostic = nativeHostDiagnostic(target);
    if (hostDiagnostic) {
        console.error(hostDiagnostic);
        exit(1);
    }

    const projectDir = resolve(parsed.dir);
    if (!statSync(projectDir, { throwIfNoEntry: false })?.isDirectory()) {
        console.error(`no project directory at ${projectDir}`);
        exit(1);
    }
    if (parsed.subcmd === "dev") {
        let viteCli: string;
        try {
            viteCli = resolveViteCli(projectDir);
        } catch (error) {
            console.error(error instanceof Error ? error.message : String(error));
            exit(1);
        }
        const status = await buildProject(projectDir, {
            target,
            release: false,
            portable: parsed.portable,
            dev: true,
        });
        if (status !== 0) exit(status);
        const server = await startViteDev(projectDir, parsed.viteArgs, viteCli);
        let appStatus: number;
        try {
            appStatus = await previewProject(projectDir, {
                target,
                portable: parsed.portable,
                devUrl: server.url,
            });
        } finally {
            await server.close();
        }
        exit(appStatus);
    }

    if (parsed.subcmd === "build") {
        exit(
            await buildProject(projectDir, {
                target,
                release: parsed.release,
                portable: parsed.portable,
                args: parsed.viteArgs,
            }),
        );
    }

    try {
        exit(
            await previewProject(projectDir, {
                target,
                release: parsed.release,
                portable: parsed.portable,
            }),
        );
    } catch (error) {
        if (!(error instanceof Error)) throw error;
        console.error(error.message);
        exit(1);
    }
}
