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

        const dispatched: number[] = [];
        const lines = await drainLog({
            dispatchWorkgroups(x) {
                dispatched.push(x);
                setTimeout(() => {
                    console.log(`${MARK}drained`, STYLE, "");
                    console.log(`${MARK}and its second line`, STYLE, "");
                }, 20);
            },
        });

        expect(dispatched).toEqual([0]);
        expect(lines).toEqual(["drained", "and its second line"]);
    });

    test("a dispatch that logs nothing returns empty at the deadline, never hangs", async () => {
        const log = globals.__gpuLog as GpuLog;
        log.lines.length = 0;
        const started = performance.now();
        const lines = await drainLog({ dispatchWorkgroups() {} }, 30);
        expect(lines).toEqual([]);
        expect(performance.now() - started).toBeGreaterThanOrEqual(25);
    });
});
