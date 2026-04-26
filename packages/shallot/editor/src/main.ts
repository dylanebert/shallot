import { mount, unmount } from "svelte";
import App from "./App.svelte";

const target = document.getElementById("app")!;
const app = mount(App, { target });

if (import.meta.hot) {
    import.meta.hot.dispose(() => {
        unmount(app);
    });
}
