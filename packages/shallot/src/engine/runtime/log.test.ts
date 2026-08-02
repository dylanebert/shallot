import { afterAll, describe, expect, test } from "bun:test";
import { captureGpuLog, drainLog, type GpuLog } from "./log";

// The console patch is process-wide and installs once, so these run in order: the opt-in check has to
// see an un-patched console. A GPU line is only distinguishable by typegpu's `%c GPU %c ` prefix —
// every log in the page arrives through the same console methods — so that marker is the contract.
const MARK = "%c GPU %c ";
const STYLE = "background: #936ff5";

const globals = globalThis as { __gpuLog?: GpuLog };
const priorLog = globals.__gpuLog;
const priorConsole = { log: console.log, warn: console.warn, error: console.error };

afterAll(() => {
    globals.__gpuLog = priorLog;
    Object.assign(console, priorConsole);
});

describe("GPU log capture", () => {
    test("no capture object means no patch — an app never pays for the affordance", () => {
        const before = console.log;
        captureGpuLog();
        expect(console.log).toBe(before);
    });

    test("GPU-marked lines are teed, page logs are not", () => {
        const log: GpuLog = { lines: [], errors: [] };
        globals.__gpuLog = log;
        captureGpuLog();

        console.log(`${MARK}hit count`, STYLE, "", 42);
        console.log("an ordinary page log");
        console.error(`${MARK}contact overflow`, STYLE, "");

        expect(log.lines).toEqual(["hit count 42", "contact overflow"]);
        expect(log.errors).toEqual(["contact overflow"]);
    });

    // typegpu's readback is fire-and-forget — it maps the ring buffer and prints from a `.then` nobody
    // can await, landing a task or more after the dispatch returns. So the fake pipeline logs
    // asynchronously: a drain that only awaited the dispatch (or the queue) reads an empty ring and
    // returns [], which is what this pins.
    test("drainLog waits out the readback gap and returns only that dispatch's lines", async () => {
        const log = globals.__gpuLog as GpuLog;
        log.lines.length = 0;
        log.errors.length = 0;
        console.log(`${MARK}from an earlier dispatch`, STYLE, "");

        let triggered = 0;
        const lines = await drainLog(() => {
            triggered++;
            setTimeout(() => {
                console.log(`${MARK}drained`, STYLE, "");
                console.log(`${MARK}and its second line`, STYLE, "");
            }, 20);
        });

        expect(triggered).toBe(1);
        expect(lines).toEqual(["drained", "and its second line"]);
    });

    test("overlapping drains serialize their triggers and keep delayed batches separate", async () => {
        const log = globals.__gpuLog as GpuLog;
        log.lines.length = 0;
        const order: string[] = [];
        const first = drainLog(() => {
            order.push("first-trigger");
            setTimeout(() => {
                order.push("first-line");
                console.log(`${MARK}first-batch`, STYLE, "");
            }, 20);
        });
        const second = drainLog(() => {
            order.push("second-trigger");
            setTimeout(() => console.log(`${MARK}second-batch`, STYLE, ""), 5);
        });

        const [firstLines, secondLines] = await Promise.all([first, second]);
        expect(firstLines).toEqual(["first-batch"]);
        expect(secondLines).toEqual(["second-batch"]);
        expect(order).toEqual(["first-trigger", "first-line", "second-trigger"]);
    });

    // Poison is intentionally monotonic for the module/page lifetime, so this must remain the final drain
    // test. Replacing the page (and therefore this module instance) is the only recovery boundary.
    test("a timeout rejects, poisons later calls before trigger, and never attributes a late line", async () => {
        const log = globals.__gpuLog as GpuLog;
        log.lines.length = 0;
        let laterTriggers = 0;
        const started = performance.now();
        const timedOut = drainLog(() => {
            setTimeout(() => console.log(`${MARK}too-late`, STYLE, ""), 30);
        }, 10);
        const queued = drainLog(() => {
            laterTriggers++;
        });

        await expect(timedOut).rejects.toThrow("capture session is poisoned");
        await expect(queued).rejects.toThrow("capture session is poisoned");
        expect(performance.now() - started).toBeGreaterThanOrEqual(8);
        expect(laterTriggers).toBe(0);

        await new Promise((resolve) => setTimeout(resolve, 30));
        expect(log.lines).toEqual(["too-late"]);
        await expect(
            drainLog(() => {
                laterTriggers++;
            }),
        ).rejects.toThrow("capture session is poisoned");
        expect(laterTriggers).toBe(0);
    });
});
