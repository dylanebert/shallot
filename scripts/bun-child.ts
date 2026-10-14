/**
 * Windows CI runs Bun from a copy named node.exe so Dawn can load. Bun named node acts as Node;
 * keep its executable and Bun host identity together when spreading into object-form spawn options.
 */
export const bunChild = (args: string[]) => ({ cmd: [process.execPath, ...args], argv0: "bun" });
