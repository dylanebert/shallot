import { afterAll, beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { publishTestChromium } from "../../scripts/publish-browser";
import { removeTestDirectory } from "../../scripts/test-directory";
import { CROSS_ORIGIN_ISOLATION } from "../project";
import { verifyPublishedGame } from "./publish-verify";

setDefaultTimeout(TIMEOUT.node);
const root = mkdtempSync(join(tmpdir(), "shallot-publish-controls-"));
const directory = join(root, "web");
mkdirSync(directory);
const script = 'dispatchEvent(new CustomEvent("shallot:startup",{detail:{kind:"app"}}));';
const html = `<link rel="icon" href="data:,"><script>${script}</script>`;
let control = "positive";
let unavailable = 0;
let requests = 0;
let unavailableStatus = 404;
let stale = 0;
let documentRequests = 0;
const SUBPATH = "/html/19938966-2182067/";
let browser: Awaited<ReturnType<typeof publishTestChromium>>;
const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    fetch(request) {
        // A build is also served from the subpath, as an HTML host's upload directory.
        const path = new URL(request.url).pathname.replace(SUBPATH, "/");
        const headers: Record<string, string> = { ...CROSS_ORIGIN_ISOLATION };
        if (path === "/" && control === "absent isolation") {
            for (const name of Object.keys(CROSS_ORIGIN_ISOLATION)) delete headers[name];
        }
        if (path === "/game.js") {
            requests++;
            if (requests <= unavailable)
                return new Response("not ready", {
                    status: unavailableStatus,
                    headers: {},
                });
            if (control === "missing asset")
                return new Response("missing", { status: 404, headers: {} });
            return new Response(readFileSync(join(directory, "game.js")), {
                headers: {
                    ...headers,
                    "Content-Type":
                        control === "wrong MIME" ? "text/plain" : "application/javascript",
                },
            });
        }
        if (path === "/" || path === "/index.html" || control === "SPA fallback") {
            documentRequests++;
            return new Response(
                documentRequests <= stale
                    ? "old build"
                    : readFileSync(join(directory, "index.html")),
                {
                    headers: {
                        ...headers,
                        "Content-Type": "text/html",
                        "Cache-Control": "no-cache",
                    },
                },
            );
        }
        return new Response("missing", { status: 404, headers: {} });
    },
});
beforeAll(async () => {
    browser = await publishTestChromium();
});
afterAll(async () => {
    await browser?.close();
    await server.stop(true);
    await removeTestDirectory(root);
});

for (const [fault, failure] of [
    ["positive", undefined],
    ["missing asset", "game.js: expected 200, got 404"],
    ["wrong MIME", "game.js: expected application/javascript, got text/plain"],
    ["SPA fallback", "A missing path answered 200, not 404"],
    ["absent isolation", "The published page is not cross-origin isolated"],
    ["no startup observation", `waitForFunction: Timeout ${TIMEOUT.cheap}ms exceeded`],
    [
        "single physics artifact before app startup",
        "Physics initialized the single artifact, not shared.",
    ],
] as const) {
    test(`the publish verifier ${failure ? `rejects ${fault}` : "accepts the isolated, observed positive control"}`, async () => {
        control = fault;
        unavailable = 0;
        requests = 0;
        const startup =
            fault === "no startup observation"
                ? ""
                : fault === "single physics artifact before app startup"
                  ? 'dispatchEvent(new CustomEvent("shallot:startup",{detail:{kind:"physics",threads:1,artifact:"single"}}));'
                  : script;
        writeFileSync(
            join(directory, "index.html"),
            `<link rel="icon" href="data:,"><script>${startup}</script>`,
        );
        writeFileSync(join(directory, "game.js"), "const game = 1;\n");
        const assets = ["index.html", "game.js"].map((path) => ({
            path,
            size: readFileSync(join(directory, path)).length,
        }));
        // The fixture dispatches startup synchronously; its missing signal must fail within the Node tier.
        const result = verifyPublishedGame(
            browser,
            new URL(SUBPATH, server.url).href,
            directory,
            assets,
            TIMEOUT.cheap,
            undefined,
            600,
        );
        if (failure)
            expect(
                await result.then(
                    () => undefined,
                    (error: Error) => error.message,
                ),
            ).toContain(failure);
        else expect(await result).toMatchObject({ hardware: false, adapter: "No GPU required" });
    });
}

test("page resource console failures name the resource URL", async () => {
    control = "positive";
    writeFileSync(
        join(directory, "index.html"),
        '<link rel="icon" href="data:,"><script src="/missing-startup.js"></script>',
    );
    expect(
        await verifyPublishedGame(browser, server.url.href, directory, [], TIMEOUT.cheap).then(
            () => undefined,
            (error: Error) => error.message,
        ),
    ).toContain(`${server.url.href}missing-startup.js`);
});

test("asset availability retries transient 404s and 5xx, but bounds persistent 404s", async () => {
    control = "positive";
    writeFileSync(join(directory, "index.html"), html);
    writeFileSync(join(directory, "game.js"), "const game = 1;\n");
    const assets = [{ path: "game.js", size: 16 }];
    for (const status of [404, 503]) {
        unavailableStatus = status;
        unavailable = 2;
        requests = 0;
        await verifyPublishedGame(
            browser,
            server.url.href,
            directory,
            assets,
            TIMEOUT.cheap,
            undefined,
            2000,
        );
        expect(requests).toBe(3);
    }
    unavailableStatus = 404;
    unavailable = Infinity;
    requests = 0;
    const start = performance.now();
    const failure = await verifyPublishedGame(
        browser,
        server.url.href,
        directory,
        assets,
        TIMEOUT.cheap,
        undefined,
        600,
    ).then(
        () => "verified",
        (error: Error) => error.message,
    );
    expect(failure).toContain(
        "Availability wait of 600 ms exhausted; game.js: expected 200, got 404",
    );
    expect(performance.now() - start).toBeLessThan(1500);
    expect(requests).toBeGreaterThanOrEqual(2);
});

test("republish retries stale index bytes and bounds a permanently wrong deploy", async () => {
    control = "positive";
    writeFileSync(join(directory, "index.html"), html);
    const assets = [{ path: "index.html", size: Buffer.byteLength(html) }];
    try {
        stale = 2;
        documentRequests = 0;
        expect(
            await verifyPublishedGame(
                browser,
                server.url.href,
                directory,
                assets,
                TIMEOUT.cheap,
                undefined,
                2000,
            ),
        ).toMatchObject({ hardware: false, adapter: "No GPU required" });
        expect(documentRequests).toBe(4); // Three byte checks, then page navigation.
        stale = Infinity;
        documentRequests = 0;
        const start = performance.now();
        const failure = await verifyPublishedGame(
            browser,
            server.url.href,
            directory,
            assets,
            TIMEOUT.cheap,
            undefined,
            600,
        ).then(
            () => "verified",
            (error: Error) => error.message,
        );
        expect(failure).toContain(
            "Availability wait of 600 ms exhausted; index.html: the deployed bytes still differ from this build",
        );
        expect(performance.now() - start).toBeLessThan(1500);
        expect(documentRequests).toBeGreaterThanOrEqual(2);
    } finally {
        stale = 0;
    }
});
