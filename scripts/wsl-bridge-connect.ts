import { bridgePrereq, start } from "./wsl-bridge";

// A thin bun-side front for `wsl-bridge.ts`, for a caller that can't be bun itself. `start()`/`bridgePrereq()`
// use `Bun.spawnSync`/`Bun.spawn` throughout, so they only run under bun — but a showcase project's own
// Playwright driver (`playwright test`, launched via its node-shebang bin) loads its config under real node,
// same as `scripts/verify.ts`'s node-bundled client for fact 2 (the bun Playwright client hangs on connect,
// real node's doesn't). This script is that node caller's one bun subprocess: probe, start, print the ws
// endpoint once, then hold the bridge open until the caller closes stdin (its teardown signal) and tear down.
//
// One line of stdout is the handshake, `BRIDGE_ENDPOINT <json>` — prefixed and scanned for (not assumed to
// be the first or last line) because the host launcher's own stdout/stderr is inherited straight through
// (`wsl-bridge.ts`'s `stdio: "inherit"`) and can interleave before or after it.

const reason = bridgePrereq();
if (reason) {
    console.log(`BRIDGE_ENDPOINT ${JSON.stringify({ skip: reason })}`);
    process.exit(0);
}

const bridge = await start();
console.log(`BRIDGE_ENDPOINT ${JSON.stringify({ connectUrl: bridge.connectUrl })}`);

await new Promise<void>((resolve) => {
    process.stdin.resume();
    process.stdin.on("end", resolve);
});
await bridge.teardown();
process.exit(0);
