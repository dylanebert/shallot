const { execFileSync } = require("node:child_process");
const { writeSync } = require("node:fs");
const option = process.argv[2] ?? "default";
const options = option === "default" ? [] : [option.includes("=") ? option : `backend=${option}`];

function mark(message) {
    writeSync(1, `[dawn-repro ${option}] ${message}\n`);
}

function inspectHost() {
    if (process.platform !== "win32") return;
    mark(`process.execPath: ${process.execPath}`);
    mark(`process.argv[0]: ${process.argv[0]}`);
    mark(`process.pid: ${process.pid}`);
    try {
        mark(`where node:\n${execFileSync("where.exe", ["node"], { encoding: "utf8" }).trim()}`);
    } catch {
        mark("where node: no matches");
    }
    try {
        const modules = execFileSync(
            "powershell.exe",
            [
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                `Get-Process -Id ${process.pid} -Module | Select-Object -ExpandProperty ModuleName`,
            ],
            { encoding: "utf8" },
        );
        for (const module of modules.trim().split(/\r?\n/)) {
            if (module) mark(`module before require: ${module}`);
        }
    } catch (error) {
        mark(`module inspection failed: ${error?.message ?? error}`);
    }
}

async function run() {
    let gpu;
    let device;
    try {
        inspectHost();
        mark("before require('webgpu')");
        const webgpu = require("webgpu");
        mark("after require('webgpu')");

        Object.assign(globalThis, webgpu.globals);
        mark(`before create(${JSON.stringify(options)})`);
        gpu = webgpu.create(options);
        mark("after create()");
        Object.defineProperty(navigator, "gpu", { configurable: true, value: gpu });

        mark("before requestAdapter()");
        const adapter = await navigator.gpu.requestAdapter();
        mark("after requestAdapter()");
        if (!adapter) throw new Error("Dawn returned no adapter");
        mark(`adapter info: ${JSON.stringify(adapter.info)}`);

        mark("before requestDevice()");
        device = await adapter.requestDevice();
        mark("after requestDevice()");
        if (!device) throw new Error("Dawn returned no device");
        mark(`device: ${device.label || "(unlabeled)"}`);

        const command = device.createCommandEncoder().finish();
        mark("before first queue.submit()");
        device.queue.submit([command]);
        mark("after first queue.submit()");
        await device.queue.onSubmittedWorkDone();
        mark("first submission completed");
    } finally {
        device?.destroy();
        if (gpu && navigator.gpu === gpu) delete navigator.gpu;
    }
}

run().catch((error) => {
    writeSync(2, `[dawn-repro ${option}] ${error?.stack ?? error}\n`);
    process.exitCode = 1;
});
