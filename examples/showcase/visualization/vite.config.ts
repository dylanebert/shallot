import { resolve } from "node:path";
import typegpu from "unplugin-typegpu/vite";

// multi-page: the gallery plus one page per demo (each iframe loads its own page, its own engine
// instance — the multi-canvas flow). Every demo HTML is a rolldown entry so the build emits them all.
// `import.meta.dirname`, not `__dirname`: vite's native config loader (the coming default) has no CJS globals.
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
        rolldownOptions: {
            input: {
                main: resolve(import.meta.dirname, "index.html"),
                immediate: resolve(import.meta.dirname, "demos/immediate.html"),
                retained: resolve(import.meta.dirname, "demos/retained.html"),
                wireframe: resolve(import.meta.dirname, "demos/wireframe.html"),
                text: resolve(import.meta.dirname, "demos/text.html"),
                animation: resolve(import.meta.dirname, "demos/animation.html"),
            },
        },
    },
};
