# multiplayer

Walk first-person's stepped route and moving lift, and push the amber box. The server owns both avatars and the box. Your avatar predicts its own movement and the lift's tick-derived trajectory; the box opts into prediction within four metres. The cyan avatar is displayed between confirmed server states. WASD moves, mouse looks, Space jumps; on a phone use the floating left stick, right drag and Jump.

## In one tab

In a created project:

```sh
bun install
bun run dev
```

Open Vite's URL. The tab runs its own CPU-only server World beside the rendered client World, connected through `seededLink`. A thin client sends a recorded walking loop and acknowledges replication, without predicting or drawing. The panel changes simulated round-trip latency and packet loss in both directions, initially 100 ms and 2%.

In the engine checkout, instead run `bunx vite` from `examples/multiplayer`; there is no example package manifest and `bun run dev` would reach the checkout's scripts.

## Two tabs, one Bun server

From the example directory, in another terminal:

```sh
bun server.ts --host 127.0.0.1 --port 3001
```

Open Vite's URL with `?server=ws://127.0.0.1:3001` in two tabs. Each connection gets one of two avatars. This mode has no thin client; the other tab supplies the remote player. The simulation controls remain visible but disabled: they belong to the in-tab link, not the socket transport.

### Remote server

For another host, run the same server file there (`--host 0.0.0.0` accepts external connections) and supply that endpoint in `?server=`. A published static client needs only the endpoint configuration; the Bun process runs separately. Use a secure WebSocket endpoint for a client served over HTTPS.

The server accepts only finite, bounded movement/look values, a sprint flag and a cumulative jump count, applied to the avatar selected by the connection—not an entity chosen by the client. Intent callbacks receive records not yet folded into the newest confirmed state, rather than a session history. The server retires accepted records behind each recipient's acknowledged tick and cumulative resolved-ID watermark; retries of retired IDs cannot apply again. Within its ID receive window, a requested tick beyond the prediction horizon is refused, resolved without application, rather than held for a later retry. The watermark stops those retries and removes refused local predictions. A client already paced by confirmations stays within that horizon; inputs recorded before its first confirmation can be refused if its clock starts far ahead. The ID receive window still bounds records held above a missing ID: later IDs outside that window remain unacknowledged until the gap is supplied. The default window fits one input and one operation per tick across the prediction horizon; this example carries its jump count within the input. Prediction retires accepted records behind its confirmed tick, keeping only unaccepted local records above the resolved-ID watermark. The avatar's input state retains its latest values when no newer record is available. This is a two-player demonstration, without authentication or persistent sessions, not a public matchmaking service. Reload after a socket disconnect to join again.

## Record a tab

Open with `?record` (or append `&record` to a `?server=` URL) to record from page load. Hold F3 and press R to download one JSON file containing the history so far. In-tab mode includes the server and client; socket mode includes only the client. Without `record`, no history is retained and the chord downloads nothing. Recording retains incoming bytes and calls in memory until the page closes; opt in only for the session you need to diagnose.

In the engine checkout, replay the downloaded file from this directory:

```sh
bun tests/replay.ts /path/to/multiplayer-recording.json
```

Replay reruns the game from creation and checks every recorded tick hash, throwing at its first differing call or hash. It then drops the client's last incoming state and checks that replay fails at that tick, so a pass is not vacuous; it prints both results. Engine and game versions must match. The browser subjects verify these Chromium recordings under Bun; a game's nonportable JavaScript math can still differ between runtimes.

## Composition

`src/network.ts` describes this game as `GAME`: shared plugins and schema, per-connection disclosure, intent validation/application, avatar assignment on join and stopped movement on leave. The extra spawns a `ClientOf` entity for each admitted connection; this game adds its avatar slot to that entity. Sockets and replication histories stay in the extra's runtime, keyed by the connection entity. Removing that entity closes its transport and discards queued bytes; World disposal releases every remaining connection. Custom transports implement `Connection.close()` with the same idempotent release contract.

The `server(GAME)` plugin composes authority into a `createApp` World. The `client(GAME, { url })` plugin connects the caller's app to a WebSocket endpoint. `await local(GAME, options)` prepares the headless server and thin-client apps and returns a plugin for the caller's primary client app, plus its transport/server pump and disposal. Child apps are prepared before the primary app builds, because app builds serialize. Client role plugins install the game's prediction at the first stepped `preFixed` boundary, after physics warm-up.

`src/runtime.ts` chooses the role, simulated faults and the thin client's walking input. `server.ts` chooses the listen address and calls `serveGame` from `multiplayer/bun`; the extra owns admission, the bounded 60 Hz tick, dead-socket cleanup and SIGTERM shutdown. The game still owns its server entry point and hosting.

## Seed search (engine checkout)

From a clean, committed engine checkout, `bun test ./scripts/multiplayer-search.oracle.ts` manually searches 20 fresh seeds. Set `SHALLOT_SEARCH_BUDGET=100` to choose another seed budget. It gates nothing. Each child runs the extra's replication subjects with seed-derived delay, loss and duplication, and this example's predicted lift rider subject with its existing 100 ms / 2% loss profile and a fresh link schedule; the existing assertions decide the result. The output names the invariants, configuration, commit and seeds run. Replication search disables loss and duplication after tick 79 before the existing drain through tick 349; convergence is checked after recovery, not under endless loss. These are simulated links, not real socket impairment, and seed count is not a coverage claim.

The first red stops the search and prints a replay command: check out its commit, then run `SHALLOT_SEARCH_COMMIT=<commit> SHALLOT_SEARCH_SEED=<seed> bun test ./scripts/multiplayer-search.oracle.ts`. Replay runs one seed. Seed plus commit identifies the schedule; the oracle refuses a dirty worktree or a mismatched replay commit. Keep the same Bun/runtime host when debugging physics-dependent example failures.

## Checks

From this directory:

```sh
bun test ./src/*.node.ts
bunx playwright install chromium
bunx playwright test
```

The Bun subjects use `createSession(GAME, options)` from `@dylanebert/shallot/multiplayer/testing` to compose caller-stepped server and client Worlds without a GPU. They run at zero delay and at 100 ms and 2% loss, keep a predicted rider on the lift, and exercise the server's trust boundary. The harness prints the session seed and its source. `options.seed` fixes it; when omitted, `createSession` reads `SHALLOT_MULTIPLAYER_SEED` as a decimal u32, defaulting to 1 if unset. For example, `SHALLOT_MULTIPLAYER_SEED=2792687340 bun test ./src/lift.node.ts` changes the lift subject's link schedule without changing its fault profile. Explicit seeds ignore the environment variable. `advance` steps a transport round; `step` advances only the named peer without advancing link time; `waitFor` checks a condition within a tick budget. Thin clients receive and acknowledge without simulating. Tests dispose the session, which checks its owned Worlds, connections and sockets after cleanup. The same harness supports real loopback WebSockets with `transport: "websocket"`; link faults do not impair that mode. Chromium runs a stepped page with input injected through `core/input`, tests the simulated link and two pages against the real Bun socket server, reads remote-avatar pixels, pushes the shared box, and checks the phone controls at 390×844. The browser config owns ports 4430 and 4431; a subject that moves the scene starts its own server with `--port 0`, which takes a free port. Bun socket prediction holds while stepped frames outrun the server's clock, so those subjects time motion in ticks, not frames.

For socket faults, pass `transport: "websocket", proxy: true`. Each ordinary or thin peer exposes `proxy.upstream` (client → server) and `proxy.downstream` (server → client), mutable `{ latency, jitter, bandwidth, stalled }` data. Times are milliseconds; bandwidth is bytes/second, zero disables the limit. Controls start disabled. Latency and uniform ±jitter are sampled per arriving TCP chunk, clamped at zero; changing them does not retime queued chunks. A stall retains bytes and releases them in order. `proxy.resetPeer()` resets both TCP legs, closing the peer; it does not clear controls or reconnect. Delay, stalls and bandwidth retain stream bytes in order; a reset ends the stream. None models packet loss beneath TCP. Delay and bandwidth use wall time while Worlds remain caller-stepped; the seed repeats jitter draws, not OS chunk boundaries or IO timing. Session disposal probes every proxy listener for connection refusal. `src/stall.node.ts` holds downstream delivery for 20 ticks and requires confirmation to catch the pre-release server tick within 40 more ticks. Chromium's socket subjects remain unimpaired.
