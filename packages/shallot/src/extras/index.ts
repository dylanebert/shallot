// the extras barrel star-exports every module — each module's index.ts owns its clean public API.
// `cells` is deliberately absent: no author-facing plugin exists yet (`extras/cells/core.ts`'s own
// header), so it has nothing for this barrel today — its substrate rides `@dylanebert/shallot/cells/core`
// instead, the tooling/custom-pipeline surface, not this one.
export * from "./gltf";
export * from "./lines";
export * from "./orbit";
export * from "./outline";
export * from "./profile";
export * from "./skin";
export * from "./sky";
export * from "./sprite";
export * from "./text";
export * from "./tween";
