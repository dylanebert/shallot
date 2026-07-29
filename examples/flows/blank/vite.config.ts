import typegpu from "unplugin-typegpu/vite";

// an ejected app owns its bundler, so it declares the TGSL build transform itself: the engine's
// shaders are transpiled at build time with no runtime fallback. Imported straight from the package —
// a vite config is loaded by node, which can't read the engine's TypeScript source. Exactly one
// instance may run; a second pass re-wraps the emitted metadata and corrupts it.
export default { plugins: [typegpu()] };
