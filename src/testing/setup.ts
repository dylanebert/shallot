import { loadNative } from "../../bin/bun-native";

await (await loadNative()).setupGlobals();
