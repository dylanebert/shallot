import { expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const ROOT = resolve(import.meta.dir, "..");

function run(command: string[], cwd: string, label: string): string {
    const result = Bun.spawnSync(command, { cwd, stdout: "pipe", stderr: "pipe" });
    const output = `${result.stdout.toString()}${result.stderr.toString()}`;
    if (result.exitCode !== 0) throw new Error(`${label} failed:\n${output}`);
    return output;
}

test("a packed headless app steps on a GPU and refuses without navigator.gpu", async () => {
    const scratch = mkdtempSync(join(tmpdir(), "shallot-packed-gpu-"));
    const project = join(scratch, "project");
    const rootPackage = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8"));
    const tarballName = `dylanebert-shallot-${rootPackage.version}.tgz`;
    const tarball = join(scratch, tarballName);
    mkdirSync(project);
    try {
        run(
            ["bun", "pm", "pack", "--ignore-scripts", "--destination", scratch, "--quiet"],
            ROOT,
            "packing Shallot",
        );
        if (!existsSync(tarball)) throw new Error(`packed tarball is missing: ${tarball}`);

        writeFileSync(
            join(project, "package.json"),
            `${JSON.stringify(
                {
                    name: "packed-headless-gpu-smoke",
                    private: true,
                    type: "module",
                    dependencies: {
                        "@dylanebert/shallot": `file:../${tarballName}`,
                        "bun-webgpu": rootPackage.devDependencies["bun-webgpu"],
                    },
                    devDependencies: { "@types/bun": rootPackage.devDependencies["@types/bun"] },
                },
                null,
                2,
            )}\n`,
        );
        writeFileSync(
            join(project, "preload.ts"),
            'import { plugin } from "bun";\nimport { shallot } from "@dylanebert/shallot/bun";\nplugin(shallot({ root: import.meta.dir }));\n',
        );
        writeFileSync(join(project, "bunfig.toml"), '[test]\npreload = ["./preload.ts"]\n');
        writeFileSync(
            join(project, "packed-engine.gpu.test.ts"),
            `import { expect, test } from "bun:test";
import { build, type Plugin } from "@dylanebert/shallot/app";
import { f32, field, Time } from "@dylanebert/shallot/ecs";
import * as Rendering from "@dylanebert/shallot/rendering";
import { drainLog, probeTexture } from "@dylanebert/shallot/runtime";
import { setupGlobals } from "bun-webgpu";

await setupGlobals();

const Ticks = { value: field(f32) };
let eid = -1;
const Counter: Plugin = {
    name: "Counter",
    components: { counter: Ticks },
    initialize(state) {
        eid = state.create();
        state.add(eid, Ticks);
        Ticks.value.set(eid, 0);
    },
    systems: [{
        group: "fixed",
        update(state) {
            for (const entity of state.query([Ticks])) {
                Ticks.value.set(entity, Ticks.value.get(entity) + 1);
            }
        },
    }],
};

test("the packed engine refuses with Bun's optional peer fix when navigator.gpu is absent", async () => {
    const previous = Object.getOwnPropertyDescriptor(navigator, "gpu");
    Object.defineProperty(navigator, "gpu", { configurable: true, value: undefined });
    try {
        await expect(build({ plugins: [Counter], defaults: false })).rejects.toThrow(
            "WebGPU unavailable: navigator.gpu is missing in Bun. Install the optional bun-webgpu peer dependency",
        );
    } finally {
        if (previous) Object.defineProperty(navigator, "gpu", previous);
        else Reflect.deleteProperty(navigator, "gpu");
    }
});

test("the packed headless plugin set steps the world through public engine subpaths", async () => {
    const app = await build({ plugins: [Counter], defaults: false });
    try {
        app.state.step(Time.FIXED_DT);
        expect(app.state.time.fixedTick).toBe(1);
        expect(app.state.only([Ticks])).toBe(eid);
        expect(Ticks.value.get(eid)).toBe(1);
        expect(Rendering.CAPTURE_CONTRACT.width).toBe(1280);
        expect(typeof Rendering.captureFrame).toBe("function");
        expect(typeof probeTexture).toBe("function");
        expect(typeof drainLog).toBe("function");
    } finally {
        app.dispose();
    }
}, 20_000);
`,
        );

        run(
            ["bun", "install", "--linker=isolated", "--no-progress"],
            project,
            "installing packed project and bun-webgpu",
        );
        expect(existsSync(join(project, "node_modules/bun-webgpu/package.json"))).toBe(true);
        const tests = run(["bun", "test"], project, "testing packed GPU project");
        expect(tests).toContain("the packed engine refuses with Bun's optional peer fix");
        expect(tests).toContain("the packed headless plugin set steps the world");
        expect(tests).toContain("2 pass");
    } finally {
        rmSync(scratch, { recursive: true, force: true });
    }
}, 300_000);
