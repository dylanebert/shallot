// Types the vendored emscripten Basis transcoder glue (basis_transcoder.js, no upstream types). The
// embind surface is untyped by nature — basis.ts aliases the whole boundary as `Embind = any`. This
// colocated declaration resolves the `.js` import for any consumer, including a workspace that checks
// shallot's source through a symlink and whose tsconfig lacks `allowJs` (orrstead).
declare const factory: any;
export default factory;
