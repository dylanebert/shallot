import { resolve } from "node:path";
import { defineConfig } from "vite";

// multi-page: the gallery plus one page per demo (each iframe loads its own page, its own engine
// instance — the multi-canvas flow). Every demo HTML is a rollup entry so the build emits them all.
export default defineConfig({
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
                tween: resolve(__dirname, "demos/tween.html"),
            },
        },
    },
});
