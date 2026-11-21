import { type App, createApp, Time } from "../../engine";
import { MultiplayerPlugin, ReplicationReceiver } from ".";
import { observeNetwork, transportClock } from "./diagnostics";
import { type Game, type PlayerId, ServerSession, server, type ThinClient } from "./game";
import { IntentSender } from "./intent";
import { type Connection, type LinkFaults, MAX_MESSAGE_BYTES, seededLink } from "./link";
import type { SessionRecorder } from "./recording";

export function createServerApp(game: Game, recorder?: SessionRecorder) {
    return createApp({
        defaults: false,
        plugins: [
            ...(recorder ? [recorder.plugin] : []),
            ...game.plugins,
            MultiplayerPlugin,
            server(game),
        ],
    });
}

interface Bot {
    advance(time: number): void;
    input(tick: number): void;
    remove(): void;
    dispose(): void;
}

/** Room authority for game-defined bot operations. Pass the requester entity supplied by
 * the server intent hook, not an ID from the operation. Only room-mode authorities have a controller. */
export const RoomBots = {
    create: (): {
        controller?: {
            canManage(requester: number): boolean;
            add(requester: number, thin: ThinClient): Promise<boolean>;
            remove(requester: number, bot: number): boolean;
        };
    } => ({}),
};

const NO_FAULTS: LinkFaults = { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 };

interface Seat<C> {
    room: Room<C>;
    eid: number;
    id: string;
    token: string;
    transport: { stream: C } & Connection;
    connection: Connection;
    expiresAt?: number;
}

interface Room<C> {
    code: string;
    private: boolean;
    seats: Map<C, number | undefined>;
    bots: Map<number, Bot>;
    buildingBots?: number;
    app?: App;
    ready?: Promise<void>;
    held?: number;
    reservation?: Reservation;
}

interface Reservation {
    creator: string;
    issuedAt: number;
}

const CODE_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";
export const ROOM_CODE_CLAIM_MS = 60_000;
const CREATOR_HOLD_MS = 30_000;
const MAX_UNCLAIMED_CODES = 128;
// Colyseus defaults to a 15 s seat reservation. Bound the time a stalled auth can hold capacity.
const AUTH_TIMEOUT_MS = 15_000;
const utf8 = new TextEncoder();
/** A seat whose oldest unconfirmed frame is this many ticks older than the current tick, counted
 * from admission or resume, is closed and leaves, or is held under Game.reconnectMs, whether or
 * not acknowledgements arrive: its sender retains every unconfirmed frame, and a client this far
 * behind re-images through resume rather than replaying. Unity Transport's 30 s disconnect
 * timeout; counting from admission spares a client still warming up after its socket opens. */
const UNCONFIRMED_SEAT_TICKS = Math.round(30 / Time.FIXED_DT);

export function roomHost<
    C extends Connection & {
        close(code?: number, reason?: string): void;
        [transportClock]?: () => number;
    },
>(
    game: Game,
    options: {
        reservationNow?: () => number;
        reconnectNow?: () => number;
        botFaults?: LinkFaults;
        sendToken(connection: C, id: string, token: string): void;
        sendRoom?(connection: C, code: string): void;
    },
    pinnedApp?: App,
    recorder?: SessionRecorder,
) {
    if (!Number.isSafeInteger(game.maxClients) || game.maxClients < 1)
        throw new Error("maxClients must be a positive integer");
    const reconnectMs = game.reconnectMs ?? 0;
    if (!Number.isFinite(reconnectMs) || reconnectMs < 0)
        throw new Error("reconnectMs must be finite and nonnegative");
    const now = options.reconnectNow ?? (() => performance.now());
    const seats = new WeakMap<C, Seat<C>>();
    const held = new Map<string, Seat<C>>();
    const humans = (room: Room<C>) => room.seats.size + (room.held ?? 0);
    const count = (room: Room<C>) => humans(room) + room.bots.size;
    const rooms = new Map<string, Room<C>>();
    const active = new Map<App, Room<C>>();
    let botTime = 0;
    const unclaimed = new Map<string, Reservation>();
    const reservationNow = options.reservationNow ?? (() => performance.now());
    const memberships = new WeakMap<C, Room<C>>();
    const authorizing = new WeakMap<C, AbortController>();
    const pinned: Room<C> | undefined = pinnedApp && {
        code: "",
        private: true,
        seats: new Map(),
        bots: new Map(),
        app: pinnedApp,
    };
    if (pinned && pinnedApp) {
        rooms.set(pinned.code, pinned);
        activate(pinned, pinnedApp);
    }
    let stopped = false;

    function expireCodes(now: number) {
        for (const [code, { issuedAt }] of unclaimed) {
            if (issuedAt + ROOM_CODE_CLAIM_MS > now) break;
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

    function createRoom(code: string, isPrivate: boolean): Room<C> {
        const room: Room<C> = { code, private: isPrivate, seats: new Map(), bots: new Map() };
        rooms.set(code, room);
        return room;
    }

    function retire(room: Room<C>, code?: number, reason?: string) {
        rooms.delete(room.code);
        for (const connection of room.seats.keys()) {
            memberships.delete(connection);
            cancelAdmission(connection);
            connection.close(code, reason);
        }
        room.seats.clear();
        for (const bot of room.bots.values()) bot.dispose();
        room.bots.clear();
        if (room.app) {
            active.delete(room.app);
            room.app.dispose();
        }
    }

    function cancelAdmission(connection: C) {
        const attempt = authorizing.get(connection);
        authorizing.delete(connection);
        attempt?.abort();
    }

    function leave(connection: C, hold = true) {
        const room = memberships.get(connection);
        if (stopped) return;
        if (!room) {
            const seat = seats.get(connection);
            if (!hold && seat?.transport.stream === connection && held.get(seat.token) === seat)
                expireSeat(seat);
            return;
        }
        memberships.delete(connection);
        const eid = room.seats.get(connection);
        room.seats.delete(connection);
        cancelAdmission(connection);
        const authority = room.app?.world.resource(ServerSession).server;
        const seat = seats.get(connection);
        if (hold && seat && authority?.peers.get(seat.eid)?.connection === seat.connection) {
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
                    (seat?.connection ?? recorder?.connection(connection) ?? connection)
            )
                authority?.disconnect(eid);
        } finally {
            if (room !== pinned && !humans(room)) retire(room);
        }
    }

    async function authorize(
        room: Room<C>,
        connection: C,
        credential: string | null,
        controller: AbortController,
    ) {
        const ownsReservation = () =>
            authorizing.get(connection) === controller &&
            memberships.get(connection) === room &&
            !connection.closed;
        if (!ownsReservation()) return;
        const timeout = setTimeout(() => controller.abort(), AUTH_TIMEOUT_MS);
        let result: Awaited<ReturnType<NonNullable<Game["onAuth"]>>> | undefined;
        try {
            const aborted = new Promise<undefined>((resolve) => {
                controller.signal.addEventListener("abort", () => resolve(undefined), {
                    once: true,
                });
            });
            result = await Promise.race([aborted, game.onAuth!(credential, controller.signal)]);
        } catch {
            // Hook errors may contain credentials. Only explicit, public refusal reasons go out.
        } finally {
            clearTimeout(timeout);
        }
        if (!ownsReservation()) return;
        authorizing.delete(connection);
        if (!controller.signal.aborted && result && "playerId" in result) {
            admit(room, connection, result.playerId);
            return;
        }
        const reason = result && "reason" in result ? result.reason : "";
        const publicReason =
            reason &&
            reason.length <= 123 &&
            utf8.encode(reason).length <= 123 &&
            (!credential || !reason.includes(credential));
        connection.close(
            1008,
            controller.signal.aborted
                ? "Admission timed out"
                : publicReason
                  ? reason
                  : "Admission refused",
        );
        leave(connection);
    }

    function admit(room: Room<C>, connection: C, playerId?: PlayerId) {
        if (connection.closed) {
            leave(connection);
            return;
        }
        try {
            const transport: Seat<C>["transport"] & { [transportClock]: () => number } = {
                stream: connection,
                [transportClock]: () => transport.stream[transportClock]?.() ?? performance.now(),
                get closed() {
                    return this.stream.closed;
                },
                close() {
                    this.stream.close();
                },
                send(bytes: Uint8Array) {
                    this.stream.send(bytes);
                },
                receive() {
                    return this.stream.receive();
                },
            };
            const eid = room
                .app!.world.resource(ServerSession)
                .server!.connect(
                    reconnectMs ? transport : connection,
                    "human",
                    undefined,
                    playerId,
                );
            if (eid === undefined) refuseFull(connection);
            else {
                room.seats.set(connection, eid);
                room.reservation = undefined;
                if (room.code) options.sendRoom?.(connection, room.code);
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

    async function start(room: Room<C>) {
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
        activate(room, app);
        if (!game.onAuth) for (const connection of room.seats.keys()) admit(room, connection);
    }

    function activate(room: Room<C>, app: App) {
        room.app = app;
        active.set(app, room);
        if (room === pinned) return;
        const authority = app.world.resource(ServerSession).server!;
        const canManage = (requester: number) => {
            const peer = authority.peers.get(requester);
            return (
                room === rooms.get(room.code) &&
                authority.leader === requester &&
                !!peer &&
                !peer.held &&
                !peer.connection.closed
            );
        };
        app.world.resource(RoomBots).controller = {
            canManage,
            async add(requester, thin) {
                if (!canManage(requester)) return false;
                const peer = authority.peers.get(requester);
                return !!(await addBot(
                    room,
                    thin,
                    () => authority.peers.get(requester) === peer && canManage(requester),
                ));
            },
            remove(requester, eid) {
                const bot = room.bots.get(eid);
                if (!bot || !canManage(requester)) return false;
                bot.remove();
                return true;
            },
        };
    }

    async function addBot(room: Room<C>, thin: ThinClient, allowed = () => true) {
        if (count(room) + (room.buildingBots ?? 0) >= game.maxClients) return;
        const link = seededLink(thin.seed, options.botFaults ?? NO_FAULTS);
        room.buildingBots = (room.buildingBots ?? 0) + 1;
        let app: App;
        try {
            app = await createApp({ defaults: false, plugins: [...game.plugins] });
        } finally {
            room.buildingBots--;
        }
        if (rooms.get(room.code) !== room || count(room) >= game.maxClients || !allowed()) {
            app.dispose();
            return;
        }
        const authority = room.app!.world.resource(ServerSession).server!;
        function dispose() {
            link.a.close();
            app.dispose();
        }
        try {
            link.advance(botTime);
            const sender = new IntentSender(link.b);
            const receiver = new ReplicationReceiver(app.world, game.schema, link.b);
            observeNetwork(app.world, link.b);
            const eid = authority.connect(link.a, "bot", thin.admit);
            if (eid !== undefined) {
                const peer = authority.peers.get(eid);
                room.bots.get(eid)?.dispose();
                const bot: Bot = {
                    advance(time) {
                        if (link.a.closed) return bot.remove();
                        link.advance(time);
                        sender.flush(receiver.flush());
                        receiver.readMessages(); // Thin clients have no game message reader.
                    },
                    input(tick) {
                        if (link.a.closed) return;
                        thin.input(app.world, sender, tick);
                        sender.flush();
                    },
                    remove() {
                        if (room.bots.get(eid) !== bot) return;
                        room.bots.delete(eid);
                        try {
                            if (authority.peers.get(eid) === peer) authority.disconnect(eid);
                        } finally {
                            dispose();
                        }
                    },
                    dispose,
                };
                room.bots.set(eid, bot);
                return { remove: bot.remove };
            }
        } catch (error) {
            dispose();
            throw error;
        }
        dispose();
    }

    /** A live seat's own credential means its client saw a drop its server has not, as after a
     * route change; hold the seat so the resume proceeds, after Colyseus's
     * checkReconnectionToken. The stale socket's later close finds no membership. */
    function replaceStaleSeat(
        token: string,
        id: string | null,
        intent: string | null,
        code: string | null,
    ) {
        for (const room of rooms.values()) {
            if (intent === "join" && code?.toUpperCase() !== room.code) continue;
            for (const connection of room.seats.keys()) {
                const seat = seats.get(connection);
                if (seat?.token !== token || seat.id !== id) continue;
                connection.close(1008, "Seat resumed by another connection");
                leave(connection);
                return;
            }
        }
    }

    function sendToken(connection: C, seat: Seat<C>) {
        options.sendToken(connection, seat.id, seat.token);
    }

    function release(seat: Seat<C>) {
        held.delete(seat.token);
        seat.room.held!--;
    }

    function expireSeat(seat: Seat<C>) {
        const authority = seat.room.app!.world.resource(ServerSession).server!;
        release(seat);
        try {
            if (authority.peers.get(seat.eid)?.connection === seat.connection)
                authority.disconnect(seat.eid);
        } finally {
            if (seat.room !== pinned && !humans(seat.room)) retire(seat.room);
        }
    }

    function expireSeats() {
        const time = now();
        for (const seat of held.values()) {
            const authority = seat.room.app!.world.resource(ServerSession).server!;
            const owns = authority.peers.get(seat.eid)?.connection === seat.connection;
            if (seat.expiresAt! > time && owns) continue;
            expireSeat(seat);
        }
    }

    function refuseFull(connection: C) {
        connection.close(1013, pinned ? "Peer limit reached" : "Room full");
    }

    function selectRoom(
        connection: C,
        intent: string | null,
        code: string | null,
        creator: string | null,
    ) {
        if (intent === "joinOrCreate" && code === null) {
            for (const room of rooms.values()) {
                if (!room.private && count(room) < game.maxClients) return room;
            }
            return createRoom(issueCode(), false);
        }
        if (intent === "join" && code) {
            const time = reservationNow();
            expireCodes(time);
            const key = code.toUpperCase();
            const room = rooms.get(key);
            const reservation = room?.reservation ?? unclaimed.get(key);
            if (
                reservation &&
                time < reservation.issuedAt + CREATOR_HOLD_MS &&
                creator !== reservation.creator
            ) {
                connection.close(4001, "First seat reserved for creator");
                return null;
            }
            if (!reservation || room) return room;
            unclaimed.delete(key);
            const claimed = createRoom(key, true);
            claimed.reservation = reservation;
            return claimed;
        }
    }

    function list() {
        return [...rooms.values()]
            .filter((room) => !room.private)
            .map((room) => ({
                code: room.code,
                clients: game.onAuth
                    ? [...room.seats.values()].filter((eid) => eid !== undefined).length +
                      (room.held ?? 0) +
                      room.bots.size
                    : count(room),
                maxClients: game.maxClients,
            }));
    }

    function reserve() {
        if (stopped) return;
        const time = reservationNow();
        expireCodes(time);
        if (unclaimed.size >= MAX_UNCLAIMED_CODES) return;
        const code = issueCode();
        const creator = crypto.randomUUID();
        unclaimed.set(code, { creator, issuedAt: time });
        return { code, creator };
    }

    function connect(
        connection: C,
        {
            intent = null,
            code = null,
            creator = null,
            resume: token = null,
            seat: id = null,
            credential = null,
        }: {
            intent?: string | null;
            code?: string | null;
            creator?: string | null;
            resume?: string | null;
            seat?: string | null;
            credential?: string | null;
        } = {},
    ) {
        if (stopped) {
            connection.close(1001, "Server stopping");
            return;
        }
        expireSeats();
        const existing = seats.get(connection);
        if (
            connection.closed ||
            memberships.has(connection) ||
            (existing?.transport.stream === connection && held.get(existing.token) === existing)
        )
            return;
        if (
            credential !== null &&
            (credential.length > MAX_MESSAGE_BYTES ||
                utf8.encode(credential).length > MAX_MESSAGE_BYTES)
        ) {
            connection.close(1008, "Invalid credential");
            return;
        }
        if (token !== null || id !== null) {
            if (token !== null && !held.has(token)) replaceStaleSeat(token, id, intent, code);
            const seat = token === null ? undefined : held.get(token);
            const authority = seat?.room.app?.world.resource(ServerSession).server;
            if (
                !seat ||
                seat.id !== id ||
                (intent === "join" && code?.toUpperCase() !== seat.room.code) ||
                authority?.peers.get(seat.eid)?.connection !== seat.connection
            ) {
                connection.close(1008, "Invalid or expired reconnection token");
                return;
            }
            release(seat);
            seat.transport.stream = connection;
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
        const room = pinned ?? selectRoom(connection, intent, code, creator);
        if (room === null) return;
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
        if (!room.app) room.ready ??= start(room);
        if (game.onAuth) {
            const attempt = new AbortController();
            authorizing.set(connection, attempt);
            const run = () => authorize(room, connection, credential, attempt);
            const ready = room.app ? run() : room.ready!.then(run);
            return { room, ready };
        }
        if (room.app) admit(room, connection);
        return { room, ready: room.ready };
    }

    function sweep() {
        for (const room of rooms.values())
            for (const peer of room.seats.keys()) if (peer.closed) leave(peer);
        expireSeats();
    }

    function advanceBots(time: number) {
        botTime = time;
        for (const { bots } of active.values()) for (const bot of bots.values()) bot.advance(time);
    }

    function closeUnconfirmedSeats() {
        for (const room of active.values()) {
            const authority = room.app!.world.resource(ServerSession).server!;
            const tick = room.app!.world.time.fixedTick;
            for (const [connection, eid] of room.seats) {
                const peer = eid === undefined ? undefined : authority.peers.get(eid);
                if (!peer || peer.held) continue;
                if (tick - Math.max(peer.sender.acknowledged, peer.since) < UNCONFIRMED_SEAT_TICKS)
                    continue;
                connection.close(1013, "Replication unacknowledged");
                leave(connection);
            }
        }
    }

    function tick() {
        closeUnconfirmedSeats();
        for (const [app, { bots }] of active) {
            for (const bot of bots.values()) bot.input(app.world.time.fixedTick);
            app.world.tick();
        }
    }

    function stop() {
        if (stopped) return;
        stopped = true;
        unclaimed.clear();
        held.clear();
        for (const room of rooms.values()) retire(room, 1001, "Server stopping");
    }
    return { connect, leave, sweep, advanceBots, tick, stop, list, reserve, addBot };
}
