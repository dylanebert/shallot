import { fileURLToPath } from "node:url";
import { shallot } from "@dylanebert/shallot/vite";

export default {
    root: fileURLToPath(new URL(".", import.meta.url)),
    plugins: [shallot()],
};
