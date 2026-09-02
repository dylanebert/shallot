import { resolve } from "node:path";
import typegpu from "unplugin-typegpu/vite";

// multi-page: the gallery plus one page per demo (each iframe loads its own page, its own engine
// instance — the multi-canvas flow). Every demo HTML is a rollup entry so the build emits them all.
export default {
    // an ejected project declares the typegpu transform itself (the engine's TGSL kernels have no
    // runtime fallback); exactly one instance may run, a second pass corrupts the metadata
    plugins: [typegpu()],
    base: "./",
    server: { port: 3000 },
    build: {
        target: "esnext",
        outDir: "dist",
        emptyOutDir: true,
        rollupOptions: {
            input: {
                main: resolve(__dirname, "index.html"),
                immediate: resolve(__dirname, "demos/immediate.html"),
                retained: resolve(__dirname, "demos/retained.html"),
                wireframe: resolve(__dirname, "demos/wireframe.html"),
                text: resolve(__dirname, "demos/text.html"),
                animation: resolve(__dirname, "demos/animation.html"),
            },
        },
    },
};
