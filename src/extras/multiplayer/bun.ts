import type { ServerWebSocket } from "bun";
import { type App, Time } from "../../engine";
import { stampReceipt } from "./diagnostics";
import { type Game, isClientMessage } from "./game";
import { type Connection, MAX_MESSAGE_BYTES } from "./link";
import { type RecordingOptions, SessionRecorder } from "./recording";
import { createServerApp, ROOM_CODE_CLAIM_MS, roomHost } from "./room-host";
import { LEAVE_CLOSE_CODE } from "./websocket";

// Paper's default packet limiter: 500 messages a second averaged over 7 seconds. A stall's
// backlog arrives at once, so the window, not the rate, decides which stalls survive.
const IngressRate = 500 / 1000;
const IngressBurst = 500 * 7;
// Bytes held between drains; without it a full burst of maximum frames holds 14 MiB. Client
// acknowledgements are cumulative and operations resend until resolved, so dropping the oldest
// costs at most stale input ticks and ping samples, and a client the server stalled stays open.
const MaxQueued = 64 * MAX_MESSAGE_BYTES;

type SocketData = { url: URL } | ServerWebSocketConnection | undefined;

/** A single accepted peer. close releases queued bytes as well as the socket. */
export class ServerWebSocketConnection implements Connection {
    private _messages: Uint8Array[] = [];
    private _closed = false;
    private _queued = 0;
    private _budget = IngressBurst;
    private _receipt = performance.now();
    readonly socket: ServerWebSocket<SocketData>;

    constructor(socket: ServerWebSocket<SocketData>) {
        this.socket = socket;
    }

    get closed(): boolean {
        return this._closed || this.socket.readyState !== 1;
    }

    close(code?: number, reason?: string): void {
        this._end(code, reason);
    }

    // Refusals bypass close, which session wrappers replace without forwarding its code.
    private _end(code?: number, reason?: string): void {
        if (this._closed) return;
        this._closed = true;
        this._messages = [];
        this._queued = 0;
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
        this._queued = 0;
        return messages;
    }

    /** @internal Copies and queues a binary frame, or closes the connection: 1003 for text,
     * 1008 past the ingress budget, 1002 when validate refuses it. Past the queue bound the
     * oldest queued messages are dropped. */
    push(bytes: Uint8Array | string, validate?: (bytes: Uint8Array) => boolean): void {
        if (this.closed) return;
        if (typeof bytes === "string") {
            this._end(1003, "Binary messages required");
            return;
        }
        const now = performance.now();
        this._budget = Math.min(IngressBurst, this._budget + (now - this._receipt) * IngressRate);
        this._receipt = now;
        if (this._budget < 1) {
            this._end(1008, "Ingress budget exceeded");
            return;
        }
        this._budget--;
        if (validate && !validate(bytes)) {
            this._end(1002, "Invalid game message");
            return;
        }
        this._queued += bytes.length;
        let dropped = 0;
        while (this._queued > MaxQueued) this._queued -= this._messages[dropped++]!.length;
        if (dropped) this._messages.splice(0, dropped);
        this._messages.push(stampReceipt(bytes.slice(), now));
    }
}

/** Self-hosted binary endpoint. Every upgrade gets a distinct connection; callbacks run once
 * at open and close. Authentication and routing belong in the application/reverse proxy.
 * Each connection may send 500 messages a second averaged over 7 seconds, in binary frames of
 * at most MAX_MESSAGE_BYTES; past 256 KiB queued between receive calls the oldest are dropped.
 * validate refuses a message before it is queued. Each refusal closes only that connection. */
export function serveWebSocket(options: {
    hostname?: string;
    port: number;
    connect: (connection: ServerWebSocketConnection, url?: URL) => void;
    validate?: (bytes: Uint8Array) => boolean;
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
            // Bun refuses a longer frame with 1009 before buffering it.
            maxPayloadLength: MAX_MESSAGE_BYTES,
            open(socket) {
                const data = socket.data;
                const connection = new ServerWebSocketConnection(socket);
                socket.data = connection;
                options.connect(connection, data && "url" in data ? data.url : undefined);
            },
            message(socket, message) {
                if (socket.data instanceof ServerWebSocketConnection)
                    socket.data.push(message, options.validate);
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
 * and JSON { code, creator } for a private room. Join first with
 * `?intent=join&code=…&creator=…`; share only the code. Until the creator is admitted
 * or 30 seconds pass, other joins close with 4001. GET /rooms returns a fresh JSON array
 * of public rooms with { code, clients, maxClients }. `clients`
 * counts occupied seats, including bots, pending startup and held reconnections. Share a private code out of
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
 * a successful resume replaces the token, and resuming a seat whose socket is still open
 * closes that socket with 1008. Invalid, expired, spent, or wrong-room
 * credentials close with 1008. Send the game credential as `?credential=…` on initial join only.
 * Use WSS and omit credential, creator and resume query
 * credentials from access logs. The first seat starts the World. Pending seats count toward maxClients but
 * are not admitted before startup succeeds. The room and its bots retire on stop or
 * when their last human seat leaves or expires. An in-flight build is not canceled, but disposes on success
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
    /** Handles HTTP requests outside /rooms that do not upgrade to WebSocket. */
    fetch?: (request: Request) => Response | undefined;
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
    const announce = new WeakSet<ServerWebSocketConnection>();
    const host = roomHost<ServerWebSocketConnection>(
        game,
        {
            ...options,
            sendRoom(connection, code) {
                if (announce.has(connection) && connection.socket.sendText(`room:${code}`) === 0)
                    connection.close(1011, "Write failed");
            },
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
        validate: isClientMessage,
        fetch(request) {
            if (pinnedApp || new URL(request.url).pathname !== "/rooms")
                return options.fetch?.(request);
            const headers = new Headers({ "Cache-Control": "no-store" });
            if (options.roomsOrigin)
                headers.set("Access-Control-Allow-Origin", options.roomsOrigin);
            if (request.method === "GET") return Response.json(host.list(), { headers });
            if (request.method !== "POST") return;
            const reservation = host.reserve();
            if (reservation) return Response.json(reservation, { status: 201, headers });
            headers.set("Retry-After", String(ROOM_CODE_CLAIM_MS / 1000));
            return new Response(null, { status: 503, headers });
        },
        connect(connection, url) {
            const query = url?.searchParams;
            if (query && query.getAll("credential").length > 1) {
                connection.close(1008, "Invalid credential");
                return;
            }
            if (query?.get("announceRoom") === "1") announce.add(connection);
            host.connect(connection, {
                intent: query?.get("intent"),
                code: query?.get("code"),
                creator: query?.get("creator"),
                resume: query?.get("resume"),
                seat: query?.get("seat"),
                credential: query?.get("credential"),
            });
        },
        disconnect(connection, code) {
            host.leave(connection, code !== LEAVE_CLOSE_CODE);
        },
    });
    const stopClock = startClock(() => {
        if (game.reconnectMs) host.sweep();
        host.advanceBots(performance.now());
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
