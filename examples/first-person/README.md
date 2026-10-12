# First person

A copyable Shallot app built around `Player`, a moving kinematic lift, and a small platforming route.

## Run

```sh
bun install
bun run dev
```

The Vite development and preview servers provide the cross-origin isolation headers used by the app. A different host needs equivalent COOP/COEP headers for multithreaded physics.

## Check

```sh
bun run test
bun run browser:install  # once for this Playwright version
bun run test:browser
```

The Node goal check runs the app's own plugins with tick-addressed `PlayerInput`, reaches the route and lift, and captures the resulting frame. The browser check builds and serves this app with its own headless Vite preview; it checks the page, focused input, and canvas presentation.
