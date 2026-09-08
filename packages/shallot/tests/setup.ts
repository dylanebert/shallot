import { loadNative } from "../../shallot-tooling/bin/bun-native";

await (await loadNative()).setupGlobals();
