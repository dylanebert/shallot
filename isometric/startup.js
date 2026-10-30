(() => {
    const bootstrap = document.querySelector("[data-shallot-bootstrap]");
    if (!(bootstrap instanceof HTMLElement)) return;

    let active = true;
    const cleanup = () => {
        if (!active) return;
        active = false;
        window.removeEventListener("error", onError, true);
        window.removeEventListener("unhandledrejection", onRejection);
        observer.disconnect();
    };
    const fail = () => {
        if (!active || !bootstrap.isConnected || bootstrap.dataset.state !== "loading") return;
        bootstrap.dataset.state = "error";
        bootstrap.setAttribute("role", "alert");
        bootstrap.setAttribute("aria-label", "Unable to start this application");
        bootstrap.replaceChildren(
            document.createTextNode("Unable to start this application. Check the console and reload."),
        );
        cleanup();
    };
    const onError = (event) => {
        if (event.target === window ||
            (event.target instanceof HTMLScriptElement && event.target.type === "module"))
            fail();
    };
    const onRejection = () => fail();
    const observer = new MutationObserver(() => {
        if (!bootstrap.isConnected) cleanup();
    });

    window.addEventListener("error", onError, true);
    window.addEventListener("unhandledrejection", onRejection);
    observer.observe(document.documentElement, { childList: true, subtree: true });
})();
