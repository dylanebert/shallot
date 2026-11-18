import type { BrowserType } from "playwright";

export interface FlySocketObservation {
    kind:
        | "socket-created"
        | "socket-first-frame"
        | "socket-closed"
        | "browser-created"
        | "browser-closed";
    id: string;
    at: number;
    url?: string;
}

type Report = (observation: FlySocketObservation) => void;

/** Keep the project's launch options; close page sockets explicitly before context teardown loses their close events. */
export function observeFlyBrowser(chromium: Pick<BrowserType, "launch">, report: Report): void {
    const launch = chromium.launch.bind(chromium);
    let browsers = 0,
        sockets = 0;
    chromium.launch = async (options) => {
        const browser = await launch(options);
        const id = `browser:${++browsers}`;
        report({ kind: "browser-created", id, at: Date.now() });
        browser.once("disconnected", () => report({ kind: "browser-closed", id, at: Date.now() }));
        const create = browser.newContext.bind(browser);
        browser.newContext = async (options) => {
            const context = await create(options);
            const closed: Promise<void>[] = [];
            await context.addInitScript(() => {
                const sockets: WebSocket[] = [];
                Object.defineProperty(window, "__shallotFlySockets", { value: sockets });
                globalThis.WebSocket = new Proxy(globalThis.WebSocket, {
                    construct(target, args) {
                        const socket = Reflect.construct(target, args) as WebSocket;
                        sockets.push(socket);
                        return socket;
                    },
                });
            });
            const close = context.close.bind(context);
            context.close = async (options) => {
                let timer: ReturnType<typeof setTimeout> | undefined;
                try {
                    await Promise.race([
                        (async () => {
                            for (const page of context.pages()) {
                                if (page.isClosed()) continue;
                                await page.evaluate(async () => {
                                    const sockets =
                                        (window as unknown as { __shallotFlySockets: WebSocket[] })
                                            .__shallotFlySockets ?? [];
                                    await Promise.all(
                                        sockets.map(
                                            (socket) =>
                                                new Promise<void>((accept) => {
                                                    if (socket.readyState === WebSocket.CLOSED) {
                                                        accept();
                                                        return;
                                                    }
                                                    socket.addEventListener(
                                                        "close",
                                                        () => accept(),
                                                        { once: true },
                                                    );
                                                    socket.close(1000, "Fly oracle cleanup");
                                                }),
                                        ),
                                    );
                                });
                            }
                            await Promise.all(closed);
                        })(),
                        new Promise<never>((_accept, reject) => {
                            timer = setTimeout(
                                () =>
                                    reject(
                                        new Error(
                                            "Browser sockets did not close within 10 seconds",
                                        ),
                                    ),
                                10_000,
                            );
                        }),
                    ]);
                } finally {
                    clearTimeout(timer);
                    await close(options);
                }
            };
            context.on("page", (page) =>
                page.on("websocket", (socket) => {
                    const id = `browser:socket:${++sockets}`;
                    report({ kind: "socket-created", id, url: socket.url(), at: Date.now() });
                    socket.once("framereceived", () =>
                        report({ kind: "socket-first-frame", id, at: Date.now() }),
                    );
                    closed.push(
                        new Promise<void>((accept) =>
                            socket.once("close", () => {
                                report({ kind: "socket-closed", id, at: Date.now() });
                                accept();
                            }),
                        ),
                    );
                }),
            );
            return context;
        };
        return browser;
    };
}

/** Observe all Bun client sockets, including close events after World disposal. */
export function observeFlyClients(report: Report, phase: string): () => void {
    const original = globalThis.WebSocket;
    let sockets = 0;
    globalThis.WebSocket = new Proxy(original, {
        construct(target, args) {
            const socket = Reflect.construct(target, args) as WebSocket;
            const id = `bun:${phase}:socket:${++sockets}`;
            report({ kind: "socket-created", id, url: socket.url, at: Date.now() });
            socket.addEventListener(
                "message",
                () => report({ kind: "socket-first-frame", id, at: Date.now() }),
                { once: true },
            );
            socket.addEventListener(
                "close",
                () => report({ kind: "socket-closed", id, at: Date.now() }),
                { once: true },
            );
            return socket;
        },
    });
    return () => {
        globalThis.WebSocket = original;
    };
}

/** A phase cannot become idle while any observed browser or socket lacks its close event. */
export function assertFlySocketsClosed(observations: FlySocketObservation[]): void {
    const pending = new Set<string>();
    for (const row of observations) {
        if (row.kind === "socket-created" || row.kind === "browser-created") {
            if (pending.has(row.id)) throw new Error(`Duplicate socket observation: ${row.id}`);
            pending.add(row.id);
        } else if (row.kind === "socket-first-frame") {
            if (!pending.has(row.id)) throw new Error(`Frame without open socket: ${row.id}`);
        } else if (!pending.delete(row.id)) throw new Error(`Close without creation: ${row.id}`);
    }
    if (pending.size) throw new Error(`Still open before idle: ${[...pending].join(", ")}`);
}
