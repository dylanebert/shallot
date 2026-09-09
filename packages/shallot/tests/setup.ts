import { loadNative } from "../../shallot-cli/bin/bun-native";

await (await loadNative()).setupGlobals();
