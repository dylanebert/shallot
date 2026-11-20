import type { ServerWebSocket } from "bun";
import { type App, createApp, Time } from "../../engine";
import { MultiplayerPlugin } from ".";
import { stampReceipt } from "./diagnostics";
import { type Game, ServerSession, server as serverRole } from "./game";
import type { Connection } from "./link";
import { type RecordingOptions, SessionRecorder } from "./recording";

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
    disconnect: (connection: ServerWebSocketConnection) => void;
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
            close(socket) {
                if (!(socket.data instanceof ServerWebSocketConnection)) return;
                socket.data.close();
                options.disconnect(socket.data);
            },
        },
    });
}

function createServerApp(game: Game, recorder?: SessionRecorder) {
    return createApp({
        defaults: false,
        plugins: [
            ...(recorder ? [recorder.plugin] : []),
            ...game.plugins,
            MultiplayerPlugin,
            serverRole(game),
        ],
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

interface Seat {
    room: Room;
    eid: number;
    id: string;
    token: string;
    transport: { socket: ServerWebSocketConnection } & Connection;
    connection: Connection;
    expiresAt?: number;
}

interface Room {
    code: string;
    private: boolean;
    seats: Map<ServerWebSocketConnection, number | undefined>;
    app?: App;
    held?: number;
}

const CODE_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";
const CLAIM_MS = 60_000;
const MAX_UNCLAIMED_CODES = 128;

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
    if (!Number.isSafeInteger(game.maxClients) || game.maxClients < 1)
        throw new Error("maxClients must be a positive integer");
    const reconnectMs = game.reconnectMs ?? 0;
    if (!Number.isFinite(reconnectMs) || reconnectMs < 0)
        throw new Error("reconnectMs must be finite and nonnegative");
    const now = options.reconnectNow ?? (() => performance.now());
    const seats = new WeakMap<ServerWebSocketConnection, Seat>();
    const held = new Map<string, Seat>();
    const count = (room: Room) => room.seats.size + (room.held ?? 0);
    const rooms = new Map<string, Room>();
    const active = new Set<App>();
    const unclaimed = new Map<string, number>();
    const reservationNow = options.reservationNow ?? (() => performance.now());
    const memberships = new WeakMap<ServerWebSocketConnection, Room>();
    const pinned: Room | undefined = pinnedApp && {
        code: "",
        private: true,
        seats: new Map(),
        app: pinnedApp,
    };
    if (pinned && pinnedApp) {
        rooms.set(pinned.code, pinned);
        active.add(pinnedApp);
    }
    let stopped = false;

    function expireCodes(now: number) {
        for (const [code, expiresAt] of unclaimed) {
            if (expiresAt > now) break;
            unclaimed.delete(code);
        }
    }

    function issueCode(): string {
        let code: string;
        do {
            const bytes = crypto.getRandomValues(new Uint8Array(10));
            code = Array.from(bytes, (byte) => CODE_ALPHABET[byte & 31]).join("");
        } while (rooms.has(code) || unclaimed.has(code));
        return code;
    }

    function createRoom(code: string, isPrivate: boolean): Room {
        const room: Room = { code, private: isPrivate, seats: new Map() };
        rooms.set(code, room);
        return room;
    }

    function retire(room: Room, code?: number, reason?: string) {
        rooms.delete(room.code);
        for (const connection of room.seats.keys()) {
            memberships.delete(connection);
            connection.close(code, reason);
        }
        room.seats.clear();
        if (room.app) {
            active.delete(room.app);
            room.app.dispose();
        }
    }

    function leave(connection: ServerWebSocketConnection) {
        const room = memberships.get(connection);
        if (!room || stopped) return;
        memberships.delete(connection);
        const eid = room.seats.get(connection);
        room.seats.delete(connection);
        const authority = room.app?.world.resource(ServerSession).server;
        const seat = seats.get(connection);
        if (seat && authority?.peers.get(seat.eid)?.connection === seat.connection) {
            authority.hold(seat.eid);
            seat.expiresAt = now() + reconnectMs;
            held.set(seat.token, seat);
            room.held = (room.held ?? 0) + 1;
            return;
        }
        try {
            if (
                eid !== undefined &&
                authority?.peers.get(eid)?.connection ===
                    (recorder?.connection(connection) ?? connection)
            )
                authority?.disconnect(eid);
        } finally {
            if (room !== pinned && !count(room)) retire(room);
        }
    }

    function admit(room: Room, connection: ServerWebSocketConnection) {
        if (connection.closed) {
            leave(connection);
            return;
        }
        try {
            const transport = {
                socket: connection,
                get closed() {
                    return this.socket.closed;
                },
                close() {
                    this.socket.close();
                },
                send(bytes: Uint8Array) {
                    this.socket.send(bytes);
                },
                receive() {
                    return this.socket.receive();
                },
            };
            const eid = room
                .app!.world.resource(ServerSession)
                .server!.connect(reconnectMs ? transport : connection);
            if (eid === undefined) refuseFull(connection);
            else {
                room.seats.set(connection, eid);
                if (reconnectMs) {
                    const token = crypto.randomUUID();
                    const seat = {
                        room,
                        eid,
                        id: crypto.randomUUID(),
                        token,
                        transport,
                        connection: recorder?.connection(transport) ?? transport,
                    };
                    seats.set(connection, seat);
                    sendToken(connection, seat);
                }
                return;
            }
        } catch (error) {
            console.error(error);
            connection.close(1011, "Admission failed");
        }
        leave(connection);
    }

    async function start(room: Room) {
        let app: App;
        try {
            app = await createServerApp(game);
        } catch (error) {
            console.error(error);
            if (rooms.get(room.code) === room) retire(room, 1011, "Room initialization failed");
            return;
        }
        if (rooms.get(room.code) !== room) {
            app.dispose();
            return;
        }
        room.app = app;
        active.add(app);
        for (const connection of room.seats.keys()) admit(room, connection);
    }

    function sendToken(connection: ServerWebSocketConnection, seat: Seat) {
        if (connection.socket.sendText(`resume:${seat.id}:${seat.token}`) === 0)
            connection.close(1011, "Write failed");
    }

    function release(seat: Seat) {
        held.delete(seat.token);
        seat.room.held!--;
    }

    function expireSeats() {
        const time = now();
        for (const seat of held.values()) {
            const authority = seat.room.app!.world.resource(ServerSession).server!;
            const owns = authority.peers.get(seat.eid)?.connection === seat.connection;
            if (seat.expiresAt! > time && owns) continue;
            release(seat);
            try {
                if (owns) authority.disconnect(seat.eid);
            } finally {
                if (seat.room !== pinned && !count(seat.room)) retire(seat.room);
            }
        }
    }

    function refuseFull(connection: ServerWebSocketConnection) {
        connection.close(1013, pinned ? "Peer limit reached" : "Room full");
    }

    function selectRoom(url?: URL) {
        const intent = url?.searchParams.get("intent");
        const code = url?.searchParams.get("code");
        if (intent === "joinOrCreate" && code === null) {
            for (const room of rooms.values()) {
                if (!room.private && count(room) < game.maxClients) return room;
            }
            return createRoom(issueCode(), false);
        }
        if (intent === "join" && code) {
            expireCodes(reservationNow());
            const key = code.toUpperCase();
            return unclaimed.delete(key) ? createRoom(key, true) : rooms.get(key);
        }
    }

    const endpoint = serveWebSocket({
        ...options,
        fetch(request) {
            if (pinned || new URL(request.url).pathname !== "/rooms") return;
            if (request.method === "GET") {
                const listing = [];
                for (const room of rooms.values()) {
                    if (room.private) continue;
                    listing.push({
                        code: room.code,
                        clients: count(room),
                        maxClients: game.maxClients,
                    });
                }
                return Response.json(listing, { headers: { "Cache-Control": "no-store" } });
            }
            if (request.method !== "POST") return;
            if (stopped) return new Response(null, { status: 503 });
            const now = reservationNow();
            expireCodes(now);
            if (unclaimed.size >= MAX_UNCLAIMED_CODES) {
                return new Response(null, {
                    status: 503,
                    headers: { "Retry-After": String(CLAIM_MS / 1000) },
                });
            }
            const code = issueCode();
            unclaimed.set(code, now + CLAIM_MS);
            return Response.json({ code }, { status: 201 });
        },
        connect(connection, url) {
            if (stopped) {
                connection.close(1001, "Server stopping");
                return;
            }
            expireSeats();
            const token = url?.searchParams.get("resume");
            const id = url?.searchParams.get("seat");
            if (token != null || id != null) {
                const seat = token == null ? undefined : held.get(token);
                const authority = seat?.room.app?.world.resource(ServerSession).server;
                if (
                    !seat ||
                    seat.id !== id ||
                    (url?.searchParams.get("intent") === "join" &&
                        url.searchParams.get("code")?.toUpperCase() !== seat.room.code) ||
                    authority?.peers.get(seat.eid)?.connection !== seat.connection
                ) {
                    connection.close(1008, "Invalid or expired reconnection token");
                    return;
                }
                release(seat);
                seat.transport.socket = connection;
                seat.room.seats.set(connection, seat.eid);
                memberships.set(connection, seat.room);
                seats.set(connection, seat);
                try {
                    authority.resume(seat.eid);
                    seat.token = crypto.randomUUID();
                    sendToken(connection, seat);
                } catch (error) {
                    console.error(error);
                    seats.delete(connection);
                    connection.close(1011, "Resumption failed");
                    try {
                        authority.disconnect(seat.eid);
                    } finally {
                        leave(connection);
                    }
                }
                return;
            }
            const room = pinned ?? selectRoom(url);
            if (!room) {
                connection.close(1008, "Unknown room or invalid intent");
                return;
            }
            if (pinned) {
                for (const peer of room.seats.keys()) if (peer.closed) leave(peer);
            }
            if (count(room) >= game.maxClients) {
                refuseFull(connection);
                return;
            }
            room.seats.set(connection, undefined);
            memberships.set(connection, room);
            if (room.app) admit(room, connection);
            else if (room.seats.size === 1) void start(room);
        },
        disconnect: leave,
    });
    const stopClock = startClock(() => {
        if (reconnectMs) {
            for (const room of rooms.values())
                for (const peer of room.seats.keys()) if (peer.closed) leave(peer);
            expireSeats();
        }
        for (const app of active) app.world.tick();
    });
    const stop = () => {
        if (stopped) return;
        stopped = true;
        stopClock();
        process.off("SIGTERM", stop);
        unclaimed.clear();
        held.clear();
        try {
            recorder?.finish();
        } finally {
            try {
                for (const room of rooms.values()) retire(room, 1001, "Server stopping");
            } finally {
                endpoint.stop(true);
            }
        }
    };
    process.on("SIGTERM", stop);
    return { port: endpoint.port, stop };
}
