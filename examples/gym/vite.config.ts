import typegpu from "unplugin-typegpu/vite";

// an ejected project owns its bundler, so it declares the typegpu transform itself — the engine's
// TGSL kernels carry no runtime fallback, and exactly one instance may run (a second pass re-wraps
// the metadata and corrupts it)
export default {
    plugins: [typegpu()],
    base: "./",
    server: {
        port: 3000,
    },
    build: {
        target: "esnext",
        outDir: "dist",
        emptyOutDir: true,
    },
};
