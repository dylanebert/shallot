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

## Composition

`src/network.ts` describes this game as `GAME`: shared plugins and schema, per-connection disclosure, intent validation/application, avatar assignment on join and stopped movement on leave. The extra spawns a `ClientOf` entity for each admitted connection; this game adds its avatar slot to that entity. Sockets and replication histories stay in the extra's runtime, keyed by the connection entity. Removing that entity closes its transport and discards queued bytes; World disposal releases every remaining connection. Custom transports implement `Connection.close()` with the same idempotent release contract.

The `server(GAME)` plugin composes authority into a `createApp` World. The `client(GAME, { url })` plugin connects the caller's app to a WebSocket endpoint. `await local(GAME, options)` prepares the headless server and thin-client apps and returns a plugin for the caller's primary client app, plus its transport/server pump and disposal. Child apps are prepared before the primary app builds, because app builds serialize. Client role plugins install the game's prediction at the first stepped `preFixed` boundary, after physics warm-up.

`src/runtime.ts` chooses the role, simulated faults and the thin client's walking input. `server.ts` chooses the listen address and calls `serveGame` from `multiplayer/bun`; the extra owns admission, the bounded 60 Hz tick, dead-socket cleanup and SIGTERM shutdown. The game still owns its server entry point and hosting.

## Checks

From this directory:

```sh
bun test ./src/*.node.ts
bunx playwright install chromium
bunx playwright test
```

The Bun subjects compose the server and two clients without a GPU, including at 100 ms and 2% loss, keep a predicted rider on the lift, and exercise the server's trust boundary. Chromium runs a stepped page with input injected through `core/input`, tests the simulated link and two pages against the real Bun socket server, reads remote-avatar pixels, pushes the shared box, and checks the phone controls at 390×844. The browser config owns ports 4430 and 4431.
