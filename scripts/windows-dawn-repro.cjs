const { writeSync } = require("node:fs");
const option = process.argv[2] ?? "default";
const options = option === "default" ? [] : [option.includes("=") ? option : `backend=${option}`];

function mark(message) {
    writeSync(1, `[dawn-repro ${option}] ${message}\n`);
}

async function run() {
    let gpu;
    let device;
    try {
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
    writeSync(2, `[dawn-repro ${backend}] ${error?.stack ?? error}\n`);
    process.exitCode = 1;
});
