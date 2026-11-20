import { type App, type Plugin, runApp } from "@dylanebert/shallot";
import { client, local } from "@dylanebert/shallot/multiplayer";
import { AppPlugin, RoomFrame } from "./app";
import { icon } from "./icons";
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
const list = element("players");
const copy = element<HTMLButtonElement>("copy");
const drop = element<HTMLButtonElement>("drop");
const leaveButton = element<HTMLButtonElement>("leave");
const enter = element<HTMLButtonElement>("enter");
const form = element<HTMLFormElement>("join");
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
let feedbackTimer: ReturnType<typeof setTimeout> | undefined;

drop.hidden = !!url;
drop.innerHTML = icon("drop");
leaveButton.innerHTML = icon("leave");
copy.insertAdjacentHTML("beforeend", icon("copy"));
enter.insertAdjacentHTML("beforeend", icon("chevron"));

// Only the code field needs focus, while the player is editing it.
const chrome = document.querySelector(".ec");
chrome?.addEventListener("mousedown", (event) => {
    if (!(event.target instanceof HTMLButtonElement)) return;
    event.preventDefault();
    codeInput.blur();
});
chrome?.addEventListener("click", (event) => {
    if (event.target instanceof HTMLButtonElement) event.target.blur();
});

enter.onclick = () => {
    form.hidden = !form.hidden;
    enter.setAttribute("aria-expanded", String(!form.hidden));
};

function closeSlot() {
    if (openSlot < 0) return;
    openSlot = -1;
    renderParty();
}

document.addEventListener("click", (event) => {
    if (!event.composedPath().includes(list)) closeSlot();
});
document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeSlot();
});

function message(text: string, transient = false) {
    clearTimeout(feedbackTimer);
    status.textContent = text;
    status.hidden = !text;
    if (transient) feedbackTimer = setTimeout(() => message(""), 2500);
}

function buttons() {
    for (const button of lobby.querySelectorAll("button")) button.disabled = busy;
    codeInput.disabled = busy;
    for (const button of list.querySelectorAll("button"))
        button.disabled = busy || dropped || rejoining || ownSlot < 0 || (!!url && !roomCode);
    copy.disabled = busy || !roomCode;
    drop.disabled = busy || rejoining || ownSlot < 0;
    drop.title = dropped ? "Resume connection" : "Drop connection";
    drop.setAttribute("aria-label", drop.title);
    drop.setAttribute("aria-pressed", String(dropped));
    leaveButton.disabled = busy;
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
    element("room-code").textContent = code;
    copy.hidden = !url || !code;
    buttons();
}

async function copyCode() {
    await navigator.clipboard.writeText(roomCode);
    closeSlot();
    message("Room code copied.", true);
}

copy.onclick = () => void act(copyCode);

function slotButton(label: string, glyph: Parameters<typeof icon>[0], text = false) {
    const button = document.createElement("button");
    button.tabIndex = -1;
    button.innerHTML = icon(glyph);
    if (text) button.append(label);
    else {
        button.title = label;
        button.setAttribute("aria-label", label);
    }
    return button;
}

function renderParty() {
    const slots: HTMLLIElement[] = [];
    for (let slot = 0; slot < GAME.maxClients; slot++) {
        const row = document.createElement("li");
        row.className = "row";
        const you = slot === ownSlot;
        const bot = bots.get(slot);
        if (you || bot || (url && roster & (1 << slot))) {
            if (openSlot === slot) openSlot = -1;
            const member = document.createElement("div");
            member.className = "member";
            member.innerHTML = bot ? icon("bot", "Bot") : icon("user", "Human player");
            const name = document.createElement("span");
            name.textContent = you ? "You" : bot ? `Bot ${bot.name}` : `Player ${slot + 1}`;
            name.title = name.textContent;
            member.append(name);
            if (you && host) member.insertAdjacentHTML("beforeend", icon("star", "Party leader"));
            if (bot) {
                const remove = slotButton(`Remove ${name.textContent}`, "remove");
                remove.onclick = () =>
                    void act(() => {
                        bot.remove();
                        bots.delete(slot);
                        renderParty();
                    });
                member.append(remove);
            }
            row.append(member);
        } else {
            row.classList.add("vacant");
            const plus = slotButton(`Open actions for seat ${slot + 1}`, "plus");
            const expanded = openSlot === slot;
            const actions = document.createElement("div");
            actions.className = "slot-actions";
            actions.id = `seat-${slot + 1}-actions`;
            actions.hidden = !expanded;
            plus.setAttribute("aria-expanded", String(expanded));
            plus.setAttribute("aria-controls", actions.id);
            plus.onclick = () => {
                openSlot = expanded ? -1 : slot;
                renderParty();
            };
            if (expanded) {
                const choice = host
                    ? slotButton("Add bot", "bot", true)
                    : slotButton("Copy code", "copy", true);
                choice.onclick = () => void act(host ? () => addBot(slot) : copyCode);
                actions.append(choice);
            }
            row.append(plus, actions);
        }
        slots.push(row);
    }
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
                    if (ownSlot >= 0) message("Rejoined as the same player.", true);
                    generation = peer.connection.generation;
                    rejoining = false;
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
    lobby.hidden = true;
    session.hidden = false;
    form.hidden = true;
    enter.setAttribute("aria-expanded", "false");
    codeInput.value = "";
    message("");
    renderParty();
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
    showCode("");
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
form.onsubmit = (event) => {
    event.preventDefault();
    codeInput.blur();
    const code = codeInput.value.trim().toUpperCase();
    if (code) void act(() => join(code));
};
leaveButton.onclick = () => void act(leave);
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
chrome?.removeAttribute("hidden");
document.querySelector("[data-shallot-bootstrap]")?.remove();
