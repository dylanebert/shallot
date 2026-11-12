import { fileURLToPath } from "node:url";
import config from "../../vite.config";

export default { ...config, root: fileURLToPath(new URL("../../", import.meta.url)) };
