// Types the vendored emscripten Draco decoder glue (draco_wasm_wrapper.js, no upstream types). The
// emscripten factory is untyped; this colocated declaration resolves the `.js` import for any consumer,
// including a workspace that checks shallot's source through a symlink and whose tsconfig lacks
// `allowJs` (orrstead).
declare const DracoDecoderModule: any;
export default DracoDecoderModule;
