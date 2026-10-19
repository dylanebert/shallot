import { BROWSER_CONFIG } from "@dylanebert/shallot/testing";
import { chromium } from "playwright";

const server = Bun.spawn(
    [process.execPath, "x", "vite", "--host", "127.0.0.1", "--port", "4175", "--strictPort"],
    { cwd: "examples/physics", stdout: "ignore", stderr: "inherit" },
);
let browser: Awaited<ReturnType<typeof chromium.launch>> | undefined;
try {
    for (;;) {
        try {
            if ((await fetch("http://127.0.0.1:4175")).ok) break;
        } catch {}
        await Bun.sleep(100);
    }
    browser = await chromium.launch(BROWSER_CONFIG.use?.launchOptions);
    const page = await browser.newPage();
    await page.route("**/src/sample.ts", async (route) => {
        const response = await route.fetch();
        const source = (await response.text())
            .replace(
                "return (scene) => {",
                "return (scene) => { const switchStart = performance.now();",
            )
            .replace(
                "const bodies = createScene(world, scene);",
                'const sceneStart = performance.now(); const bodies = createScene(world, scene); performance.measure("oracle:scene", {start: sceneStart, end: performance.now()});',
            )
            .replace(
                "return bodies;",
                'performance.measure("oracle:replace", {start: switchStart, end: performance.now()}); return bodies;',
            );
        await route.fulfill({ response, body: source });
    });
    await page.addInitScript(() => {
        const gpu = navigator.gpu;
        const requestAdapter = gpu.requestAdapter.bind(gpu);
        gpu.requestAdapter = async (...args) => {
            const adapter = await requestAdapter(...args);
            if (adapter) {
                const requestDevice = adapter.requestDevice.bind(adapter);
                adapter.requestDevice = async (...args) => {
                    const start = performance.now();
                    const device = await requestDevice(...args);
                    performance.measure("oracle:device", { start, end: performance.now() });
                    return device;
                };
            }
            return adapter;
        };
    });
    await page.goto("http://127.0.0.1:4175");
    await page.getByRole("button", { name: "Scene", exact: true }).waitFor();
    await page.locator("[data-shallot-loading]").waitFor({ state: "detached" });
    await page.evaluate(() => performance.clearMeasures());
    const start = performance.now();
    await page.getByRole("button", { name: "Scene", exact: true }).click();
    await page.getByRole("button", { name: "Bridge", exact: true }).click();
    await page.getByRole("slider", { name: "Gravity scale" }).waitFor();
    await page.locator("[data-shallot-loading]").waitFor({ state: "detached" });
    console.log(
        JSON.stringify(
            {
                switchMs: performance.now() - start,
                measures: await page.evaluate(() =>
                    performance
                        .getEntriesByType("measure")
                        .map(({ name, startTime, duration }) => ({ name, startTime, duration })),
                ),
            },
            null,
            2,
        ),
    );
} finally {
    await browser?.close();
    server.kill();
    await server.exited;
}
