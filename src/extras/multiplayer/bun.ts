import type { ServerWebSocket } from "bun";
import { type App, Time } from "../../engine";
import { stampReceipt } from "./diagnostics";
import type { Game } from "./game";
import type { Connection } from "./link";
import { type RecordingOptions, SessionRecorder } from "./recording";
import { createServerApp, ROOM_CODE_CLAIM_MS, roomHost } from "./room-host";
import { LEAVE_CLOSE_CODE } from "./websocket";

type SocketData = { url: URL } | ServerWebSocketConnection | undefined;

/** A single accepted peer. close releases queued bytes as well as the socket. */
export class ServerWebSocketConnection implements Connection {
    private _messages: Uint8Array[] = [];
    private _closed = false;
    readonly socket: ServerWebSocket<SocketData>;

    constructor(socket: ServerWebSocket<SocketData>) {
        this.socket = socket;
    }

    get closed(): boolean {
        return this._closed || this.socket.readyState !== 1;
    }

    close(code?: number, reason?: string): void {
        if (this._closed) return;
        this._closed = true;
        this._messages = [];
        this.socket.close(code, reason);
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
    connect: (connection: ServerWebSocketConnection, url?: URL) => void;
    disconnect: (connection: ServerWebSocketConnection, code: number) => void;
    fetch?: (request: Request) => Response | undefined;
}) {
    return Bun.serve<SocketData>({
        hostname: options.hostname,
        port: options.port,
        fetch(request, server) {
            if (server.upgrade(request, { data: { url: new URL(request.url) } })) return;
            return (
                options.fetch?.(request) ??
                new Response("WebSocket upgrade required", { status: 426 })
            );
        },
        websocket: {
            open(socket) {
                const data = socket.data;
                const connection = new ServerWebSocketConnection(socket);
                socket.data = connection;
                options.connect(connection, data && "url" in data ? data.url : undefined);
            },
            message(socket, message) {
                if (typeof message === "string") {
                    socket.close(1003, "Binary messages required");
                    return;
                }
                if (socket.data instanceof ServerWebSocketConnection) socket.data.push(message);
            },
            close(socket, code) {
                if (!(socket.data instanceof ServerWebSocketConnection)) return;
                socket.data.close();
                options.disconnect(socket.data, code);
            },
        },
    });
}

function startClock(tick: () => void) {
    let previous = performance.now();
    let accumulator = 0;
    const timer = setInterval(() => {
        const now = performance.now();
        accumulator += Math.min((now - previous) / 1000, Time.MAX_FIXED_STEPS * Time.FIXED_DT);
        previous = now;
        while (accumulator >= Time.FIXED_DT) {
            tick();
            accumulator -= Time.FIXED_DT;
        }
    }, Time.FIXED_DT * 1000);
    return () => clearInterval(timer);
}

/** Self-hosts one headless authoritative game. Owns admission, a 60 Hz clock with at most four
 * catch-up ticks per callback, dead-peer cleanup and SIGTERM shutdown. stop is idempotent and
 * releases the timer, sockets, signal listener and World. The caller chooses address and hosting. */
export async function serveGame(
    game: Game,
    options: Omit<RoomHostOptions, "reservationNow"> & { recording?: RecordingOptions },
) {
    const recorder = options.recording && new SessionRecorder(game, "server", options.recording);
    const app = await createServerApp(game, recorder);
    recorder?.attachServer();
    try {
        const host = hostRooms(game, options, app, recorder);
        app.world.onDispose(host.stop);
        return { ...host, world: app.world };
    } catch (error) {
        app.dispose();
        throw error;
    }
}

/** Hosts independent authoritative Worlds of one game. POST /rooms returns 201
 * and JSON { code } for a private room. GET /rooms returns a fresh JSON array
 * of public rooms with { code, clients, maxClients }. `clients`
 * counts occupied seats, including pending startup and held reconnections. Share a private code out of
 * band and claim it within 60 seconds with `?intent=join&code=…`.
 * A code is 10 case-insensitive
 * Crockford Base32 characters (no I, L, O or U). Codes are unique among live
 * rooms and may be randomly issued again after retirement. The host holds at
 * most 128 unclaimed codes to allow concurrent invitations without unbounded
 * idle records. At capacity, POST returns 503 with Retry-After: 60 (seconds).
 * Claim or expiry frees a slot; expired codes are pruned on admission requests,
 * not on the game clock.
 * A socket cannot create a private room with a client-supplied code.
 * `?intent=joinOrCreate` fills an available public room before making another.
 * Refusals send no game image: 1013 means full, 1008 means unknown/expired code
 * or invalid intent, and 1011 means server initialization failed.
 * With `game.reconnectMs`, admission sends a WebSocket text control message
 * `resume:<seat>:<token>` beside the binary stream. Resume with `?seat=…&resume=…`;
 * a successful resume replaces the token. Invalid, expired, spent, or wrong-room
 * credentials close with 1008. Use WSS and omit resume query
 * credentials from access logs. The first seat starts the World. Pending seats count toward maxClients but
 * are not admitted before startup succeeds. Last leave or stop retires the room
 * immediately; an in-flight build is not canceled, but disposes on success
 * without admitting anyone. A rejected build cleans up its partial World and
 * retires the code, closing any seats still waiting. stop is idempotent. */
export function serveRooms(game: Game, options: RoomHostOptions) {
    return hostRooms(game, options);
}

interface RoomHostOptions {
    hostname?: string;
    port: number;
    /** Allowed browser origin for GET and POST /rooms. Omit for same-origin access. */
    roomsOrigin?: string;
    /** Test-only monotonic time in milliseconds for code expiry; defaults to performance.now. */
    reservationNow?: () => number;
    /** Monotonic time in milliseconds for held seats. */
    reconnectNow?: () => number;
}

function hostRooms(
    game: Game,
    options: RoomHostOptions,
    pinnedApp?: App,
    recorder?: SessionRecorder,
) {
    const host = roomHost<ServerWebSocketConnection>(
        game,
        {
            ...options,
            sendToken(connection, id, token) {
                if (connection.socket.sendText(`resume:${id}:${token}`) === 0)
                    connection.close(1011, "Write failed");
            },
        },
        pinnedApp,
        recorder,
    );
    const endpoint = serveWebSocket({
        ...options,
        fetch(request) {
            if (pinnedApp || new URL(request.url).pathname !== "/rooms") return;
            const headers = new Headers({ "Cache-Control": "no-store" });
            if (options.roomsOrigin) headers.set("Access-Control-Allow-Origin", options.roomsOrigin);
            if (request.method === "GET") return Response.json(host.list(), { headers });
            if (request.method !== "POST") return;
            const code = host.reserve();
            if (code) return Response.json({ code }, { status: 201, headers });
            headers.set("Retry-After", String(ROOM_CODE_CLAIM_MS / 1000));
            return new Response(null, { status: 503, headers });
        },
        connect(connection, url) {
            const query = url?.searchParams;
            host.connect(connection, {
                intent: query?.get("intent"),
                code: query?.get("code"),
                resume: query?.get("resume"),
                seat: query?.get("seat"),
            });
        },
        disconnect(connection, code) {
            host.leave(connection, code !== LEAVE_CLOSE_CODE);
        },
    });
    const stopClock = startClock(() => {
        if (game.reconnectMs) host.sweep();
        host.tick();
    });
    let stopped = false;
    const stop = () => {
        if (stopped) return;
        stopped = true;
        stopClock();
        process.off("SIGTERM", stop);
        try {
            recorder?.finish();
        } finally {
            try {
                host.stop();
            } finally {
                endpoint.stop(true);
            }
        }
    };
    process.on("SIGTERM", stop);
    return { port: endpoint.port, stop };
}
