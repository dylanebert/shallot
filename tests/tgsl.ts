import { plugin } from "bun";
import typegpu from "unplugin-typegpu/bun";

// TGSL function bodies are transpiled at load time — typegpu parses nothing at runtime, so without
// this transform every kernel resolves with no metadata and CPU-called kernels return NaN. Bun
// exposes it as a loader plugin, which only takes effect from a preload (bunfig.toml `[test]`), and
// exactly one instance may run: a second pass re-wraps the emitted metadata and corrupts it.
// `.ts` only: the bun arm re-emits every file its filter matches with an explicit loader, even the
// ones it prunes, which strips CJS default-export interop from a plain `.js` dependency (picomatch,
// under vite, dies at import). TGSL is only ever authored in TypeScript here.
plugin(typegpu({ include: /\.tsx?$/ }));
