import { resolve } from "node:path";
import config from "../../vite.config";

export default { ...config, root: resolve(import.meta.dirname, "../..") };
