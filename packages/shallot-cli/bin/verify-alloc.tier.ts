import { expect, test } from "bun:test";
import { mkdirSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { chromium } from "playwright";
import { REAL_GPU_LAUNCH } from "../src/harness/browser";
import { isSoftwareAdapter } from "./verify";

const TRACE_TIMEOUT_MS = 20_000;
const CHURN_CAP_BYTES = 64 * 1024 * 1024;
const CHURN_CAP_MS = 5_000;
const INLINE_FIXTURE = `<!doctype html>
<meta charset="utf-8">
<title>s5b allocation observer premise</title>
<body><main>allocation observer premise</main></body>`;

type RecordValue = Record<string, unknown>;
type Cdp = {
    send(method: string, params?: unknown): Promise<unknown>;
    once(event: string, listener: (payload: unknown) => void): void;
};
type TraceEvent = {
    name?: string;
    cat?: string;
    ph?: string;
    ts?: number;
    dur?: number;
    pid?: number;
    tid?: number;
    args?: RecordValue;
};
type TraceReceipt = {
    label: string;
    requestedCategories: string[];
    completion: RecordValue;
    bytes: number;
    events: TraceEvent[];
    startMarker: string;
    endMarker: string;
    startFence?: TraceEvent;
    endFence?: TraceEvent;
    collections: Collection[];
    unknownCollectionEvents: TraceEvent[];
};
type Collection = TraceEvent & {
    collectionClass: "young" | "major";
    isolateValue: string;
    inWindow: boolean;
};

type SessionReceipt = {
    sourceRef: string;
    command: string;
    launch: typeof REAL_GPU_LAUNCH;
    userAgent: string;
    hardware: string;
    target: RecordValue;
    isolateId: string;
    categories: string[];
    requestedGcCategories: string[];
    traces: TraceReceipt[];
    churn: { allocatedBytes: number; elapsedMs: number; rounds: number };
};

let markerCounter = 0;

function record(value: unknown): RecordValue {
    if (!value || typeof value !== "object" || Array.isArray(value)) return {};
    return value as RecordValue;
}

function stringValue(value: unknown): string | undefined {
    return typeof value === "string" || typeof value === "number" ? String(value) : undefined;
}

function includesMarker(value: unknown, marker: string): boolean {
    if (typeof value === "string") return value === marker || value.includes(marker);
    if (Array.isArray(value)) return value.some((item) => includesMarker(item, marker));
    if (value && typeof value === "object")
        return Object.values(value).some((item) => includesMarker(item, marker));
    return false;
}

function eventEnd(event: TraceEvent): number | undefined {
    if (typeof event.ts !== "number") return undefined;
    return event.ts + (typeof event.dur === "number" ? Math.max(0, event.dur) : 0);
}

function inWindow(event: TraceEvent, start: TraceEvent, end: TraceEvent): boolean {
    const eventStart = event.ts;
    const eventFinish = eventEnd(event);
    const startTs = start.ts;
    const endTs = end.ts;
    return (
        typeof eventStart === "number" &&
        typeof eventFinish === "number" &&
        typeof startTs === "number" &&
        typeof endTs === "number" &&
        eventStart >= startTs &&
        eventFinish <= endTs
    );
}

function collectionClass(event: TraceEvent): "young" | "major" | undefined {
    const args = record(event.args);
    const labels = [
        stringValue(event.name),
        stringValue(args.type),
        stringValue(args.gcType),
        stringValue(args.gc_type),
        stringValue(args.reason),
        stringValue(args.gc_reason),
    ]
        .filter((value): value is string => value != null)
        .map((value) => value.toLowerCase());
    if (labels.some((value) => /scavenge|minor|young/.test(value))) return "young";
    if (labels.some((value) => /mark|major|full|compact/.test(value))) return "major";
    return undefined;
}

function isolateValue(event: TraceEvent): string | undefined {
    const args = record(event.args);
    for (const key of ["isolate", "isolateId", "isolate_id"]) {
        const value = stringValue(args[key]);
        if (value != null) return value;
    }
    return undefined;
}

function findFence(events: TraceEvent[], marker: string): TraceEvent | undefined {
    return events.find((event) => includesMarker(event.args, marker));
}

function traceEvents(value: unknown): TraceEvent[] {
    if (Array.isArray(value)) return value.filter((event): event is TraceEvent => !!event);
    const root = record(value);
    return Array.isArray(root.traceEvents)
        ? root.traceEvents.filter((event): event is TraceEvent => !!event)
        : [];
}

async function readTrace(cdp: Cdp, stream: string): Promise<{ text: string; bytes: number }> {
    const chunks: Buffer[] = [];
    let eof = false;
    while (!eof) {
        const response = record(await cdp.send("IO.read", { handle: stream }));
        const data = typeof response.data === "string" ? response.data : "";
        chunks.push(
            response.base64Encoded === true
                ? Buffer.from(data, "base64")
                : Buffer.from(data, "utf8"),
        );
        eof = response.eof === true;
    }
    await cdp.send("IO.close", { handle: stream });
    const output = Buffer.concat(chunks);
    return { text: output.toString("utf8"), bytes: output.byteLength };
}

async function marker(
    cdp: Cdp,
    page: { evaluate(fn: (value: string) => unknown, value: string): Promise<unknown> },
    label: string,
): Promise<string> {
    const id = `s5b-${label}-${++markerCounter}`;
    await cdp.send("Tracing.recordClockSyncMarker", { syncId: id });
    await page.evaluate((value) => console.timeStamp(value), id);
    return id;
}

async function stopTrace(cdp: Cdp): Promise<RecordValue> {
    const completed = new Promise<unknown>((resolve) => {
        cdp.once("Tracing.tracingComplete", resolve);
    });
    await cdp.send("Tracing.end");
    return record(
        await Promise.race([
            completed,
            new Promise((_, reject) =>
                setTimeout(
                    () => reject(new Error("Tracing.tracingComplete did not arrive")),
                    TRACE_TIMEOUT_MS,
                ),
            ),
        ]),
    );
}

async function traceBracket(
    cdp: Cdp,
    page: {
        evaluate(fn: (value: string) => unknown, value: string): Promise<unknown>;
        waitForTimeout(ms: number): Promise<void>;
    },
    label: string,
    categories: string[],
    action: () => Promise<unknown>,
    isolateId: string,
): Promise<TraceReceipt> {
    await cdp.send("Tracing.start", {
        categories: ["-*", ...categories].join(","),
        transferMode: "ReturnAsStream",
        streamFormat: "json",
        streamCompression: "none",
    });
    const startMarker = await marker(cdp, page, `${label}-start`);
    await action();
    const endMarker = await marker(cdp, page, `${label}-end`);
    const completion = await stopTrace(cdp);
    const stream = stringValue(completion.stream);
    if (!stream) throw new Error(`${label}: tracingComplete did not provide a stream`);
    if (completion.dataLossOccurred === true) throw new Error(`${label}: trace data loss occurred`);
    const raw = await readTrace(cdp, stream);
    const events = traceEvents(JSON.parse(raw.text));
    const startFence = findFence(events, startMarker);
    const endFence = findFence(events, endMarker);
    if (!startFence || !endFence)
        throw new Error(`${label}: start/end trace fences were not retained`);
    if (typeof startFence.ts !== "number" || typeof endFence.ts !== "number")
        throw new Error(`${label}: trace fences have no timestamps`);
    const candidates = events.filter((event) =>
        categories.some((category) => (event.cat ?? "").split(",").includes(category)),
    );
    const collections: Collection[] = [];
    const unknownCollectionEvents: TraceEvent[] = [];
    for (const event of candidates) {
        const value = isolateValue(event);
        const kind = collectionClass(event);
        if (kind == null) {
            if (event.name?.toLowerCase().includes("gc")) unknownCollectionEvents.push(event);
            continue;
        }
        if (value == null) throw new Error(`${label}: collection event has no isolate identity`);
        if (value !== isolateId)
            throw new Error(`${label}: collection event crossed isolate identity`);
        collections.push({
            ...event,
            collectionClass: kind,
            isolateValue: value,
            inWindow: inWindow(event, startFence, endFence),
        });
    }
    return {
        label,
        requestedCategories: categories,
        completion,
        bytes: raw.bytes,
        events,
        startMarker,
        endMarker,
        startFence,
        endFence,
        collections,
        unknownCollectionEvents,
    };
}

function writeReceipt(receiptDir: string, receipt: Partial<SessionReceipt>): void {
    mkdirSync(receiptDir, { recursive: true });
    writeFileSync(`${receiptDir}/premise-receipt.json`, `${JSON.stringify(receipt, null, 2)}\n`);
}

async function fixtureServer(): Promise<{ url: string; close(): Promise<void> }> {
    const server = createServer((_request, response) => {
        response.writeHead(200, { "content-type": "text/html; charset=utf-8" });
        response.end(INLINE_FIXTURE);
    });
    await new Promise<void>((resolve, reject) => {
        server.once("error", reject);
        server.listen(0, "127.0.0.1", resolve);
    });
    const address = server.address();
    if (!address || typeof address === "string") throw new Error("fixture server has no port");
    return {
        url: `http://127.0.0.1:${address.port}/`,
        close: async () => new Promise((resolve) => server.close(() => resolve())),
    };
}

async function adapterIdentity(page: {
    evaluate<T>(fn: () => Promise<T>): Promise<T>;
}): Promise<{ userAgent: string; hardware: string }> {
    return page.evaluate(async () => {
        const adapter = await navigator.gpu?.requestAdapter();
        const info = adapter?.info;
        return {
            userAgent: navigator.userAgent,
            hardware:
                [info?.vendor, info?.architecture, info?.device, info?.description]
                    .filter(Boolean)
                    .join(" / ") || "unknown",
        };
    });
}

async function churn(page: {
    evaluate<T>(
        fn: (limits: { cap: number; ms: number }) => T,
        limits: { cap: number; ms: number },
    ): Promise<T>;
}): Promise<SessionReceipt["churn"]> {
    return page.evaluate(
        ({ cap, ms }) => {
            const started = performance.now();
            let allocatedBytes = 0;
            let rounds = 0;
            while (allocatedBytes < cap && performance.now() - started < ms) {
                const batch = new Array<{ value: number; text: string }>(4096);
                for (let i = 0; i < batch.length; i++)
                    batch[i] = { value: i, text: `${rounds}:${i}` };
                allocatedBytes += batch.length * 32;
                rounds++;
            }
            return { allocatedBytes, elapsedMs: performance.now() - started, rounds };
        },
        { cap: CHURN_CAP_BYTES, ms: CHURN_CAP_MS },
    );
}

test("G5: one real CDP tracing premise session observes complete GC controls", async () => {
    const receiptDir =
        process.env.S5B_RECEIPT_DIR ?? `/tmp/shallot-cut-the-gate-s5b-premise-${Date.now()}`;
    const server = await fixtureServer();
    const browser = await chromium.launch({ headless: false, ...REAL_GPU_LAUNCH });
    const context = await browser.newContext();
    const page = await context.newPage();
    const cdp = (await context.newCDPSession(page)) as unknown as Cdp;
    let partial: Partial<SessionReceipt> = {
        sourceRef: "e68a1058d1d99ff314eebdd61bd0a7ba19161502",
        command:
            "SHALLOT_DISPLAY_REQUIRED=1 bun test --timeout 120000 ./packages/shallot-cli/bin/verify-alloc.tier.ts",
        launch: REAL_GPU_LAUNCH,
        traces: [],
    };
    try {
        await page.goto(server.url, { waitUntil: "load", timeout: 30_000 });
        const identity = await adapterIdentity(page);
        if (isSoftwareAdapter(identity.hardware))
            throw new Error(`unavailable real adapter: ${identity.hardware}`);
        await cdp.send("Runtime.enable");
        const target = record(record(await cdp.send("Target.getTargetInfo")).targetInfo);
        const isolateId = stringValue(record(await cdp.send("Runtime.getIsolateId")).id);
        if (!target.targetId || !isolateId) throw new Error("target/isolate identity unavailable");
        const categories = record(await cdp.send("Tracing.getCategories")).categories;
        if (!Array.isArray(categories))
            throw new Error("Tracing.getCategories returned no category list");
        const available = categories.filter(
            (category): category is string => typeof category === "string",
        );
        const requestedGcCategories = available.filter((category) => /v8.*gc/i.test(category));
        if (requestedGcCategories.length === 0)
            throw new Error("no actual V8 GC tracing category is available");
        const requestedCategories = [
            ...new Set([
                ...requestedGcCategories,
                ...available.filter((category) => category === "blink.console"),
            ]),
        ];
        partial = {
            ...partial,
            userAgent: identity.userAgent,
            hardware: identity.hardware,
            target,
            isolateId,
            categories: available,
            requestedGcCategories,
            traces: [],
        };
        const traces = [
            await traceBracket(
                cdp,
                page,
                "quiet",
                requestedCategories,
                async () => {
                    await page.waitForTimeout(250);
                },
                isolateId,
            ),
            await traceBracket(
                cdp,
                page,
                "forced",
                requestedCategories,
                async () => {
                    await cdp.send("HeapProfiler.collectGarbage");
                },
                isolateId,
            ),
            await traceBracket(
                cdp,
                page,
                "churn",
                requestedCategories,
                async () => {
                    partial.churn = await churn(page);
                },
                isolateId,
            ),
        ];
        partial.traces = traces;
        const allocationChurn = partial.churn;
        if (!allocationChurn) throw new Error("young-generation churn did not return controls");
        expect(allocationChurn.allocatedBytes).toBeGreaterThan(0);
        expect(allocationChurn.allocatedBytes).toBeLessThanOrEqual(CHURN_CAP_BYTES);
        expect(allocationChurn.elapsedMs).toBeLessThanOrEqual(CHURN_CAP_MS + 250);
        for (const trace of traces) {
            expect(trace.bytes).toBeGreaterThan(0);
            expect(trace.completion.dataLossOccurred).toBe(false);
            expect(trace.collections.every((event) => event.inWindow)).toBe(true);
            expect(trace.unknownCollectionEvents).toEqual([]);
        }
        const quiet = traces[0];
        const forced = traces[1];
        const churnTrace = traces[2];
        expect(quiet.collections).toHaveLength(0);
        expect(forced.collections.length).toBeGreaterThan(0);
        expect(churnTrace.collections.some((event) => event.collectionClass === "young")).toBe(
            true,
        );
        expect(
            traces.some((trace) =>
                trace.collections.some((event) => event.collectionClass === "major"),
            ),
        ).toBe(true);
        expect(
            traces.some((trace) =>
                trace.collections.some((event) => event.collectionClass === "young"),
            ),
        ).toBe(true);
        writeReceipt(receiptDir, partial);
    } finally {
        writeReceipt(receiptDir, partial);
        await context.close().catch(() => {});
        await browser.close().catch(() => {});
        await server.close().catch(() => {});
    }
}, 120_000);
