import { type App, type Plugin, runApp } from "@dylanebert/shallot";
import { client, local } from "@dylanebert/shallot/multiplayer";
import { AppPlugin, RoomFrame } from "./app";
import { ClientSession, GAME, PLAYER_NAMES } from "./network";

function element<T extends HTMLElement = HTMLElement>(id: string) {
    return document.getElementById(id) as T;
}

const status = element("status");
const codeInput = element<HTMLInputElement>("code");
const lobby = element("lobby");
const session = element("session");
const players = element("players");
const listing = element("listing");
const botButton = element<HTMLButtonElement>("bot");
const removeButton = element<HTMLButtonElement>("remove");
const dropButton = element<HTMLButtonElement>("drop");
const server = new URLSearchParams(location.search).get("server");
const url = server ? new URL(server, location.href) : undefined;
if (url && !["http:", "https:", "ws:", "wss:"].includes(url.protocol))
    throw new Error("Use an HTTP or WebSocket server address");
const http = url && new URL("/rooms", url);
if (http)
    http.protocol = url?.protocol === "wss:" || url?.protocol === "https:" ? "https:" : "http:";
const host = url
    ? undefined
    : await local(GAME, {
          rooms: true,
          seed: 812,
          faults: { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 },
      });
type Seat = Awaited<ReturnType<NonNullable<typeof host>["join"]>>;
let app: App | undefined;
let seat: Seat | undefined;
let bots: Awaited<ReturnType<Seat["addBot"]>>[] = [];
let busy = false;
let dropped = false;
let rejoining = false;
let roster = 0;
let shownRoster = -1;
let shownGeneration: number | undefined;
let joined = false;
let seed = 0;

element("mode").textContent = url ? "BUN SERVER" : "IN THIS TAB";
element("dev").hidden = !host;

function message(text: string) {
    status.textContent = text;
}

async function roomsRequest(method: "GET" | "POST") {
    const response = await fetch(http!, { method });
    if (!response.ok) throw new Error(`Rooms server returned ${response.status}.`);
    return response;
}

async function list() {
    const rooms: { code: string; clients: number; maxClients: number }[] = host
        ? host.list()
        : await (await roomsRequest("GET")).json();
    if (lobby.hidden) return;
    const items = rooms.map(({ code, clients, maxClients }) => {
        const li = document.createElement("li");
        const button = document.createElement("button");
        button.textContent = code;
        button.setAttribute("aria-label", `Join room ${code}`);
        button.onclick = () => void act(() => join(code));
        const seats = document.createElement("span");
        seats.textContent = `${clients} / ${maxClients} players`;
        li.append(button, seats);
        return li;
    });
    if (!items.length) {
        const li = document.createElement("li");
        li.textContent = "No public rooms yet.";
        items.push(li);
    }
    if (listing.textContent !== items.map((item) => item.textContent).join(""))
        listing.replaceChildren(...items);
}

function buttons() {
    for (const button of lobby.querySelectorAll("button")) button.disabled = busy;
    botButton.disabled = busy || dropped || rejoining || roster === (1 << GAME.maxClients) - 1;
    removeButton.disabled = busy || dropped || rejoining || bots.length === 0;
    dropButton.disabled = busy || rejoining || !joined;
    dropButton.textContent = dropped ? "Resume connection" : "Drop connection";
    element<HTMLButtonElement>("leave").disabled = busy;
}

async function act(action: () => Promise<void> | void) {
    if (busy) return;
    busy = true;
    buttons();
    try {
        await action();
    } catch (error) {
        message(error instanceof Error ? error.message : String(error));
    } finally {
        busy = false;
        buttons();
    }
}

const RoomPanel: Plugin = {
    name: "RoomPanel",
    recovery: "stateless",
    systems: [
        {
            name: "show-room-roster",
            group: "simulation",
            update(world) {
                const peer = world.resource(ClientSession).client;
                if (!peer || peer.slot < 0 || peer.prediction.receiver.confirmed < 0) return;
                const generation = peer.connection.generation;
                if (!joined || generation !== shownGeneration) {
                    joined = true;
                    message(generation ? "Rejoined as the same player." : "Connected.");
                    shownGeneration = generation;
                    rejoining = false;
                    buttons();
                }
                roster = peer.roster;
                if (roster === shownRoster) return;
                shownRoster = roster;
                players.replaceChildren(
                    ...PLAYER_NAMES.flatMap((name, slot) => {
                        if (!(roster & (1 << slot))) return [];
                        const li = document.createElement("li");
                        const label = document.createElement("span");
                        label.textContent = name;
                        li.append(label);
                        const role = slot === peer.slot ? "You" : host ? "Bot" : "Player";
                        const badge = document.createElement("span");
                        badge.textContent = role;
                        li.append(badge);
                        return [li];
                    }),
                );
                buttons();
            },
        },
    ],
};

async function join(code?: string) {
    message("Joining…");
    let plugin: Plugin;
    if (url) {
        const endpoint = new URL(url);
        endpoint.protocol = ["https:", "wss:"].includes(url.protocol) ? "wss:" : "ws:";
        endpoint.searchParams.set("intent", code ? "join" : "joinOrCreate");
        if (code) endpoint.searchParams.set("code", code);
        plugin = client(GAME, {
            url: endpoint.href,
            onError: () => {
                leave();
                message("The room closed or refused the connection. Check the code and try again.");
            },
        });
    } else {
        seat = await host!.join(code);
        plugin = seat.plugin;
    }
    try {
        app = await runApp({
            plugins: [plugin, AppPlugin, RoomPanel],
            setup(world) {
                world.resource(RoomFrame).pump = host?.pump;
            },
        });
    } catch (error) {
        seat?.leave();
        seat = undefined;
        throw error;
    }
    const roomCode = seat?.code ?? code;
    element("room-code").textContent = roomCode ? `Room code: ${roomCode}` : "Public room";
    lobby.hidden = true;
    session.hidden = false;
}

function leave() {
    for (const bot of bots) bot.remove();
    bots = [];
    if (document.pointerLockElement) document.exitPointerLock();
    app?.dispose();
    app = undefined;
    seat = undefined;
    joined = dropped = rejoining = false;
    roster = 0;
    shownRoster = -1;
    shownGeneration = undefined;
    players.replaceChildren();
    session.hidden = true;
    lobby.hidden = false;
    message("Left the room.");
    void list().catch(() => {});
}

element("quick").onclick = () => void act(() => join());
element("create").onclick = () =>
    void act(async () => {
        const code = host ? host.create() : (await (await roomsRequest("POST")).json()).code;
        if (!code) throw new Error("No private room codes are available yet.");
        await join(code);
    });
element<HTMLFormElement>("join").onsubmit = (event) => {
    event.preventDefault();
    const code = codeInput.value.trim().toUpperCase();
    if (code) void act(() => join(code));
};
element("leave").onclick = () => void act(leave);
botButton.onclick = () =>
    void act(async () => {
        if (!seat) return;
        bots.push(
            await seat.addBot({
                seed: ++seed,
                input(_world, sender, tick) {
                    const direction = Math.floor(tick / 90) % 2 ? 0.35 : -0.35;
                    sender.record(tick + 1, "input", [direction, 0, 0, 0, 0, 0]);
                },
            }),
        );
    });
removeButton.onclick = () => void act(() => bots.pop()?.remove());
dropButton.onclick = () =>
    void act(() => {
        if (!seat) return;
        if (!dropped) {
            seat.cut();
            dropped = true;
            message("Connection dropped. Resume within 10 seconds.");
        } else if (seat.resume()) {
            dropped = false;
            rejoining = true;
            message("Rejoining…");
        } else {
            leave();
            message("Your seat expired. Join again to play.");
        }
    });

const poll = () => {
    if (!lobby.hidden)
        void list().catch(() => message("Cannot list rooms. Is the Bun server running?"));
};
const polling = setInterval(poll, 2000);
window.addEventListener("pagehide", (event) => {
    if (event.persisted) return;
    clearInterval(polling);
    app?.dispose();
    host?.dispose();
});
message("Join a public room or share a private code.");
document.querySelector(".ec.controls")?.removeAttribute("hidden");
document.querySelector("[data-shallot-bootstrap]")?.remove();
poll();
