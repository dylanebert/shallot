import type { ServerWebSocket } from "bun";
import { createApp, Time } from "../../engine";
import { MultiplayerPlugin } from ".";
import { stampReceipt } from "./diagnostics";
import { type Game, ServerSession, server as serverRole } from "./game";
import type { Connection } from "./link";
import { type RecordingOptions, SessionRecorder } from "./recording";

/** A single accepted peer. close releases queued bytes as well as the socket. */
export class ServerWebSocketConnection implements Connection {
    private _messages: Uint8Array[] = [];
    private _closed = false;
    readonly socket: ServerWebSocket<ServerWebSocketConnection | undefined>;

    constructor(socket: ServerWebSocket<ServerWebSocketConnection | undefined>) {
        this.socket = socket;
    }

    get closed(): boolean {
        return this._closed || this.socket.readyState !== 1;
    }

    close(): void {
        if (this._closed) return;
        this._closed = true;
        this._messages = [];
        this.socket.close();
    }

    /** Bun owns queued writes; a dropped write closes the peer rather than silently losing bytes.
     * Writes to a closing or closed peer are discarded, so one peer cannot throw into a server tick;
     * `disconnect` reports its close. */
    send(bytes: Uint8Array): void {
        if (this.closed) return;
        if (this.socket.sendBinary(bytes) === 0) this.socket.close(1011, "Write failed");
    }

    receive(): Uint8Array[] {
        const messages = this._messages;
        this._messages = [];
        return messages;
    }

    /** @internal */
    push(bytes: Uint8Array): void {
        if (!this.closed) this._messages.push(stampReceipt(bytes.slice(), performance.now()));
    }
}

/** Self-hosted binary endpoint. Every upgrade gets a distinct connection; callbacks run once
 * at open and close. Authentication and routing belong in the application/reverse proxy. */
export function serveWebSocket(options: {
    hostname?: string;
    port: number;
    connect: (connection: ServerWebSocketConnection) => void;
    disconnect: (connection: ServerWebSocketConnection) => void;
}) {
    return Bun.serve<ServerWebSocketConnection | undefined>({
        hostname: options.hostname,
        port: options.port,
        fetch(request, server) {
            if (server.upgrade(request, { data: undefined })) return;
            return new Response("WebSocket upgrade required", { status: 426 });
        },
        websocket: {
            open(socket) {
                const connection = new ServerWebSocketConnection(socket);
                socket.data = connection;
                options.connect(connection);
            },
            message(socket, message) {
                if (typeof message === "string") {
                    socket.close(1003, "Binary messages required");
                    return;
                }
                socket.data!.push(message);
            },
            close(socket) {
                socket.data!.close();
                options.disconnect(socket.data!);
            },
        },
    });
}

/** Self-hosts one headless authoritative game. Owns admission, a 60 Hz clock with at most four
 * catch-up ticks per callback, dead-peer cleanup and SIGTERM shutdown. stop is idempotent and
 * releases the timer, sockets, signal listener and World. The caller chooses address and hosting. */
export async function serveGame(
    game: Game,
    options: { hostname?: string; port: number; recording?: RecordingOptions },
) {
    const recorder = options.recording && new SessionRecorder(game, "server", options.recording);
    const app = await createApp({
        defaults: false,
        plugins: [
            ...(recorder ? [recorder.plugin] : []),
            ...game.plugins,
            MultiplayerPlugin,
            serverRole(game),
        ],
    });
    recorder?.attachServer();
    const authority = app.world.resource(ServerSession).server!;
    const entities = new WeakMap<ServerWebSocketConnection, number>();
    let endpoint: ReturnType<typeof serveWebSocket>;
    try {
        endpoint = serveWebSocket({
            ...options,
            connect(connection) {
                const eid = authority.connect(connection);
                if (eid === undefined) connection.socket.close(1013, "Peer limit reached");
                else entities.set(connection, eid);
            },
            disconnect(connection) {
                const eid = entities.get(connection);
                if (
                    eid !== undefined &&
                    authority.peers.get(eid)?.connection ===
                        (recorder?.connection(connection) ?? connection)
                )
                    authority.disconnect(eid);
                entities.delete(connection);
            },
        });
    } catch (error) {
        app.dispose();
        throw error;
    }
    let previous = performance.now(),
        accumulator = 0,
        stopped = false;
    const timer = setInterval(() => {
        const now = performance.now();
        accumulator += Math.min((now - previous) / 1000, Time.MAX_FIXED_STEPS * Time.FIXED_DT);
        previous = now;
        while (accumulator >= Time.FIXED_DT) {
            app.world.tick();
            accumulator -= Time.FIXED_DT;
        }
    }, Time.FIXED_DT * 1000);
    const stop = () => {
        if (stopped) return;
        stopped = true;
        clearInterval(timer);
        process.off("SIGTERM", stop);
        try {
            recorder?.finish();
        } finally {
            try {
                endpoint.stop(true);
            } finally {
                app.dispose();
            }
        }
    };
    app.world.onDispose(stop);
    process.on("SIGTERM", stop);
    return { port: endpoint.port, world: app.world, stop };
}
