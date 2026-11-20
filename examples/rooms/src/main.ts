import { type App, type Plugin, runApp } from "@dylanebert/shallot";
import { client, local } from "@dylanebert/shallot/multiplayer";
import { AppPlugin, RoomFrame } from "./app";
import { ClientSession, GAME } from "./network";

function element<T extends HTMLElement = HTMLElement>(id: string) {
    return document.getElementById(id) as T;
}

declare const ROOMS_SAME_ORIGIN: boolean;

const page = new URL(location.href);
const server = page.searchParams.get("server");
const url = server
    ? new URL(server, page)
    : typeof ROOMS_SAME_ORIGIN !== "undefined" && ROOMS_SAME_ORIGIN
      ? new URL(page.origin)
      : undefined;
if (url && !["http:", "https:", "ws:", "wss:"].includes(url.protocol))
    throw new Error("Use an HTTP or WebSocket server address");
const host = url
    ? undefined
    : await local(GAME, {
          rooms: true,
          seed: 812,
          faults: { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 },
      });

type Seat = Awaited<ReturnType<NonNullable<typeof host>["join"]>>;
type Bot = Awaited<ReturnType<Seat["addBot"]>> & { name: number };
const bots = new Map<number, Bot>();
const lobby = element("lobby");
const session = element("session");
const status = element("status");
const panel = element<HTMLDetailsElement>("room-panel");
const list = element("players");
const copy = element<HTMLButtonElement>("copy");
const drop = element<HTMLButtonElement>("drop");
const codeLabel = element("room-code");
const count = element("party-count");
const crown = element("party-crown");
const codeInput = element<HTMLInputElement>("code");
let app: App | undefined;
let seat: Seat | undefined;
let roomCode = "";
let busy = false;
let dropped = false;
let rejoining = false;
let ownSlot = -1;
let roster = 0;
let generation: number | undefined;
let openSlot = -1;
let seed = 0;

drop.hidden = !!url;

function message(text: string) {
    status.textContent = text;
    status.hidden = !text;
}

function buttons() {
    for (const button of lobby.querySelectorAll("button")) button.disabled = busy;
    codeInput.disabled = busy;
    for (const button of list.querySelectorAll("button"))
        button.disabled = busy || dropped || rejoining || ownSlot < 0;
    copy.disabled = busy || !roomCode;
    drop.disabled = busy || rejoining || ownSlot < 0;
    drop.textContent = dropped ? "Resume connection" : "Drop connection";
    element<HTMLButtonElement>("leave").disabled = busy;
}

async function act(action: () => Promise<void> | void) {
    if (busy) return;
    busy = true;
    message("");
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

function showCode(code: string) {
    roomCode = code;
    codeLabel.textContent = code ? `Room code ${code}` : "";
    element("share-code").textContent = code;
    copy.textContent = "⎘";

    buttons();
}

copy.onclick = () =>
    void act(async () => {
        await navigator.clipboard.writeText(roomCode);
        copy.textContent = "✓";
    });

function slotButton(text: string, action: (button: HTMLButtonElement) => Promise<void> | void) {
    const button = document.createElement("button");
    button.textContent = text;
    button.onclick = () => void act(() => action(button));
    return button;
}

function renderParty() {
    let occupied = 0;
    const slots: HTMLLIElement[] = [];
    for (let slot = 0; slot < GAME.maxClients; slot++) {
        const row = document.createElement("li");
        const you = slot === ownSlot;
        const bot = bots.get(slot);
        if (you || bot || (url && roster & (1 << slot))) {
            occupied++;
            const portrait = document.createElement("span");
            portrait.className = "portrait";
            portrait.textContent = you ? (host ? "Y ★" : "Y") : bot ? "B" : "P";
            portrait.setAttribute("aria-hidden", "true");
            const name = document.createElement("span");
            name.textContent = you ? "You" : bot ? `Bot ${bot.name}` : `Player ${slot + 1}`;
            if (you && host) name.setAttribute("aria-label", "You, party leader");
            row.append(portrait, name);
            if (bot) {
                const remove = slotButton("×", () => {
                    bot.remove();
                    bots.delete(slot);
                    renderParty();
                });
                remove.setAttribute("aria-label", `Remove Bot ${bot.name}`);
                row.append(remove);
            }
        } else {
            row.className = "empty";
            if (host) {
                const action =
                    openSlot === slot
                        ? slotButton("Add bot", () => addBot(slot))
                        : slotButton("", () => {
                              openSlot = slot;
                              renderParty();
                          });
                if (openSlot !== slot) action.setAttribute("aria-label", `Empty slot ${slot + 1}`);
                row.append(action);
            }
        }
        slots.push(row);
    }
    count.textContent = `${occupied} / ${GAME.maxClients}`;
    list.replaceChildren(...slots);
    buttons();
}

async function addBot(slot: number) {
    if (!seat) return;
    const used = new Set([...bots.values()].map((bot) => bot.name));
    let name = 1;
    while (used.has(name)) name++;
    const bot = await seat.addBot({
        seed: ++seed,
        input(_world, sender, tick) {
            const direction = Math.floor(tick / 90) % 2 ? 0.35 : -0.35;
            sender.record(tick + 1, "input", [direction, 0, 0, 0, 0, 0]);
        },
    });
    bots.set(slot, { ...bot, name });
    openSlot = -1;
    renderParty();
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
                if (peer.connection.generation !== generation) {
                    generation = peer.connection.generation;
                    rejoining = false;
                    message("Rejoined as the same player.");
                    buttons();
                }
                if (peer.slot !== ownSlot || peer.roster !== roster) {
                    ownSlot = peer.slot;
                    roster = peer.roster;
                    renderParty();
                }
            },
        },
    ],
};

async function join(code?: string) {
    message("Joining…");
    let plugin: Plugin;
    let refused = false;
    showCode("");
    if (url) {
        const endpoint = new URL(url);
        endpoint.protocol = ["https:", "wss:"].includes(url.protocol) ? "wss:" : "ws:";
        endpoint.searchParams.set("intent", code ? "join" : "joinOrCreate");
        if (code) endpoint.searchParams.set("code", code);
        plugin = client(GAME, {
            url: endpoint.href,
            onRoom: showCode,
            onError() {
                refused = true;
                if (app) leave();
                message("The room closed or refused the connection. Check the code and try again.");
            },
        });
    } else {
        seat = await host!.join(code);
        plugin = seat.plugin;
        showCode(seat.code);
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
    if (refused) {
        leave();
        throw new Error("The room refused the connection. Check the code and try again.");
    }
    element("rooms-title").textContent = "Party";
    count.hidden = codeLabel.hidden = false;
    crown.hidden = !host;
    lobby.hidden = true;
    session.hidden = false;
    message("");
    renderParty();
    if (matchMedia("(max-width: 480px)").matches) panel.open = false;
}

function leave() {
    for (const bot of bots.values()) bot.remove();
    bots.clear();
    if (document.pointerLockElement) document.exitPointerLock();
    app?.dispose();
    app = undefined;
    seat = undefined;
    dropped = rejoining = false;
    ownSlot = openSlot = -1;
    roster = 0;
    generation = undefined;
    list.replaceChildren();
    session.hidden = true;
    lobby.hidden = false;
    element("rooms-title").textContent = "Rooms";
    codeLabel.hidden = count.hidden = crown.hidden = true;
    panel.open = true;
    message("");
}

element("quick").onclick = () => void act(() => join());
element("create").onclick = () =>
    void act(async () => {
        let code: string | undefined;
        if (host) code = host.create();
        else {
            const endpoint = new URL("/rooms", url);
            endpoint.protocol =
                url?.protocol === "wss:" || url?.protocol === "https:" ? "https:" : "http:";
            const response = await fetch(endpoint, { method: "POST" });
            if (!response.ok) throw new Error(`Rooms server returned ${response.status}.`);
            ({ code } = await response.json());
        }
        if (!code) throw new Error("No private room codes are available yet.");
        await join(code);
    });
element<HTMLFormElement>("join").onsubmit = (event) => {
    event.preventDefault();
    const code = codeInput.value.trim().toUpperCase();
    if (code) void act(() => join(code));
};
element("leave").onclick = () => void act(leave);
drop.onclick = () =>
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

window.addEventListener("pagehide", (event) => {
    if (event.persisted) return;
    app?.dispose();
    host?.dispose();
});
document.querySelector(".ec.controls")?.removeAttribute("hidden");
document.querySelector("[data-shallot-bootstrap]")?.remove();
