// `import.meta.env` is read by the engine's own source (`view.ts`'s `devEnabled`), and the package
// ships `.ts` — a consumer's tsc typechecks the dependency. `@types/node` (a runtime dep) provides
// `process` and `node:worker_threads` but not `import.meta.env` (a Vite/bundler convention), so the
// package ships this ambient augmentation. Referenced from the entrypoint (`src/index.ts`) so a
// consumer's program picks it up with no consumer config.
interface ImportMeta {
    readonly env?: Record<string, unknown>;
}
