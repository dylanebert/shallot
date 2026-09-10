import { loadNative } from "../engine/runtime/bun-native";

await (await loadNative()).setupGlobals();
