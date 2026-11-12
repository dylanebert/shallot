/**
 * Every Shallot Rust kernel links with this linear-memory maximum. WebAssembly exports return
 * pointers as signed 32-bit values; under 2 GiB every address is a positive one, so JavaScript
 * can read it unconverted, and an allocation past the cap fails as null rather than wrapping.
 * Emscripten's default `MAXIMUM_MEMORY` is 2 GiB for the same reason.
 */
export const WASM_MAX_MEMORY = 2 ** 31;

/** The rustc flag that links a kernel's memory with `WASM_MAX_MEMORY` as its maximum. */
export const WASM_MEMORY_FLAG = `-C link-arg=--max-memory=${WASM_MAX_MEMORY}`;
