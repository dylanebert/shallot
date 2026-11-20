import { type App, type Plugin, runApp } from "@dylanebert/shallot";
import { client, local, RoomBot, RoomLeader } from "@dylanebert/shallot/multiplayer";
import { AppPlugin, EntryView, RoomFrame } from "./app";
import { icon } from "./icons";
import { ClientSession, ConnectionSlot, GAME, thinBot } from "./network";

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
const copyFeedback = element("copy-feedback");
const drop = element<HTMLButtonElement>("drop");
const leaveButton = element<HTMLButtonElement>("leave");
const create = element<HTMLButtonElement>("create");
const form = element<HTMLFormElement>("join");
const codeInput = element<HTMLInputElement>("code");
let app: App | undefined;
let entry: App | undefined;
let seat: Seat | undefined;
let roomCode = "";
let busy = false;
let dropped = false;
let rejoining = false;
let ownSlot = -1;
let roster = 0;
let botSlots = 0;
let leader: number | undefined;
let generation: number | undefined;
let openSlot = -1;
let seed = 0;
let feedbackTimer: ReturnType<typeof setTimeout> | undefined;
let resetCopy: (() => void) | undefined;

drop.hidden = !!url;
drop.innerHTML = icon("drop");
leaveButton.innerHTML = icon("leave");
copy.insertAdjacentHTML("beforeend", icon("copy"));
create.insertAdjacentHTML("afterbegin", icon("lock"));

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

function closeSlot() {
    if (openSlot < 0) return;
    openSlot = -1;
    renderParty();
}

document.addEventListener("click", (event) => {
    if (
        !event
            .composedPath()
            .some((node) => node instanceof Element && node.matches(".slot-actions, .row > button"))
    )
        closeSlot();
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
    if (roomCode !== code) resetCopy?.();
    roomCode = code;
    element("room-code").textContent = code;
    copy.hidden = !url || !code;
    buttons();
}

async function copyCode(button: HTMLButtonElement, label: string) {
    await navigator.clipboard.writeText(roomCode);
    resetCopy?.();
    const glyph = button.querySelector("svg")!;
    glyph.outerHTML = icon("check");
    const confirmation = "Room code copied";
    button.setAttribute("aria-label", confirmation);
    button.title = confirmation;
    copyFeedback.textContent = confirmation;
    const timer = setTimeout(() => resetCopy?.(), 1500);
    resetCopy = () => {
        clearTimeout(timer);
        button.querySelector("svg")?.replaceWith(glyph);
        button.setAttribute("aria-label", label);
        button.title = label;
        copyFeedback.textContent = "";
        resetCopy = undefined;
    };
}

copy.onclick = () => void act(() => copyCode(copy, "Copy room code"));

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

function requestBot(slot: number, remove = false) {
    const peer = app?.world.resource(ClientSession).client;
    if (peer) peer.requestBot(peer.prediction.receiver.confirmed, slot, remove);
}

function renderParty() {
    const canManage = ownSlot === leader;
    const slots: HTMLLIElement[] = [];
    for (let slot = 0; slot < GAME.maxClients; slot++) {
        const row = document.createElement("li");
        row.className = "row";
        const you = slot === ownSlot;
        const localBot = bots.get(slot);
        const bot = url ? botSlots & (1 << slot) : localBot;
        const occupied = you || bot || (url && roster & (1 << slot));
        let trigger: HTMLButtonElement | undefined;
        if (occupied) {
            const member =
                bot && canManage
                    ? slotButton(`Open actions for seat ${slot + 1}`, "bot")
                    : document.createElement("div");
            member.className = "member";
            member.innerHTML = bot ? icon("bot", "Bot") : icon("user", "Human player");
            const name = document.createElement("span");
            name.textContent = you
                ? "You"
                : bot
                  ? `Bot ${localBot?.name ?? slot + 1}`
                  : `Player ${slot + 1}`;
            name.title = name.textContent;
            member.append(name);
            if (slot === leader)
                member.insertAdjacentHTML("beforeend", icon("star", "Party leader"));
            row.append(member);
            if (member instanceof HTMLButtonElement) trigger = member;
        } else {
            row.classList.add("vacant");
            trigger = slotButton(`Open actions for seat ${slot + 1}`, "plus");
            row.append(trigger);
        }
        if (trigger) {
            const expanded = openSlot === slot;
            const actions = document.createElement("div");
            actions.className = "slot-actions";
            actions.id = `seat-${slot + 1}-actions`;
            actions.setAttribute("role", "group");
            actions.setAttribute("aria-label", `Seat ${slot + 1} actions`);
            actions.hidden = !expanded;
            trigger.setAttribute("aria-expanded", String(expanded));
            trigger.setAttribute("aria-controls", actions.id);
            trigger.onclick = () => {
                openSlot = expanded ? -1 : slot;
                renderParty();
            };
            if (expanded && canManage) {
                const choice = slotButton(
                    bot ? "Remove bot" : "Add bot",
                    bot ? "remove" : "bot",
                    true,
                );
                choice.onclick = () =>
                    void act(() => {
                        if (!bot && host) return addBot(slot);
                        if (localBot) {
                            localBot.remove();
                            bots.delete(slot);
                            closeSlot();
                        } else {
                            closeSlot();
                            requestBot(slot, !!bot);
                        }
                    });
                actions.append(choice);
            }
            if (expanded && url && !occupied) {
                const choice = slotButton("Copy code", "copy", true);
                choice.title = "Copy code";
                choice.onclick = () => void act(() => copyCode(choice, "Copy code"));
                actions.append(choice);
            }
            row.append(actions);
        } else if (openSlot === slot) openSlot = -1;
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
    const bot = await seat.addBot(thinBot(++seed, slot));
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
                const [leaderEid] = world.query([RoomLeader, ConnectionSlot]);
                const leaderSlot =
                    leaderEid === undefined
                        ? undefined
                        : world.get(leaderEid, ConnectionSlot)?.slot;
                let kinds = 0;
                if (url) {
                    for (const eid of world.query([RoomBot, ConnectionSlot]))
                        kinds |= 1 << world.get(eid, ConnectionSlot)!.slot;
                }
                if (
                    peer.slot !== ownSlot ||
                    peer.roster !== roster ||
                    leaderSlot !== leader ||
                    kinds !== botSlots
                ) {
                    ownSlot = peer.slot;
                    roster = peer.roster;
                    leader = leaderSlot;
                    botSlots = kinds;
                    renderParty();
                }
            },
        },
    ],
};

async function join(code?: string, creator?: string) {
    message("Joining…");
    let plugin: Plugin;
    let refused = false;
    showCode("");
    if (url) {
        const endpoint = new URL(url);
        endpoint.protocol = ["https:", "wss:"].includes(url.protocol) ? "wss:" : "ws:";
        endpoint.searchParams.set("intent", code ? "join" : "joinOrCreate");
        if (code) endpoint.searchParams.set("code", code);
        if (creator) endpoint.searchParams.set("creator", creator);
        plugin = client(GAME, {
            url: endpoint.href,
            onRoom: showCode,
            onError() {
                refused = true;
                const problem =
                    "The room closed or refused the connection. Check the code and try again.";
                if (app)
                    void leave().then(
                        () => message(problem),
                        () => message(problem),
                    );
                else message(problem);
            },
        });
    } else {
        seat = await host!.join(code, creator);
        plugin = seat.plugin;
        showCode(seat.code);
    }
    entry?.dispose();
    entry = undefined;
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
        await showEntry();
        throw error;
    }
    if (refused) {
        await leave();
        throw new Error("The room refused the connection. Check the code and try again.");
    }
    lobby.hidden = true;
    session.hidden = false;
    codeInput.value = "";
    message("");
    renderParty();
}

async function showEntry() {
    entry = await runApp({ plugins: [EntryView] });
}

async function leave() {
    for (const bot of bots.values()) bot.remove();
    bots.clear();
    if (document.pointerLockElement) document.exitPointerLock();
    app?.dispose();
    app = undefined;
    seat = undefined;
    dropped = rejoining = false;
    ownSlot = openSlot = -1;
    roster = botSlots = 0;
    leader = undefined;
    generation = undefined;
    list.replaceChildren();
    session.hidden = true;
    showCode("");
    message("");
    await showEntry();
    lobby.hidden = false;
}

element("quick").onclick = () => void act(() => join());
create.onclick = () =>
    void act(async () => {
        let reservation: { code: string; creator: string } | undefined;
        if (host) reservation = host.create();
        else {
            const endpoint = new URL("/rooms", url);
            endpoint.protocol =
                url?.protocol === "wss:" || url?.protocol === "https:" ? "https:" : "http:";
            const response = await fetch(endpoint, { method: "POST" });
            if (!response.ok) throw new Error(`Rooms server returned ${response.status}.`);
            reservation = await response.json();
        }
        if (!reservation) throw new Error("No private room codes are available yet.");
        await join(reservation.code, reservation.creator);
    });
form.onsubmit = (event) => {
    event.preventDefault();
    codeInput.blur();
    const code = codeInput.value.trim().toUpperCase();
    if (code) void act(() => join(code));
};
leaveButton.onclick = () => void act(leave);
drop.onclick = () =>
    void act(async () => {
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
            await leave();
            message("Your seat expired. Join again to play.");
        }
    });

window.addEventListener("pagehide", (event) => {
    if (event.persisted) return;
    entry?.dispose();
    app?.dispose();
    host?.dispose();
});
await showEntry();
chrome?.removeAttribute("hidden");
document.querySelector("[data-shallot-bootstrap]")?.remove();
