# Project Structure

3D game engine with Unity DOTS-inspired architecture.

## Stack

- Runtime: Bun/Node.js
- Language: TypeScript 5.6
- Physics: Rapier 3D WASM
- Build: Vite 5.4 with TypeScript declarations

## Commands

- Dev: `bun run dev` (watch mode build)
- Build: `bun run build` (production build)
- Type Check: `bun run check` (TypeScript validation)
- Example: `bun run example` (run demo application)
- Format: `bun run format` (Prettier code formatting)
- Lint: `bun run lint` (ESLint code analysis)

## Layout

```
shalloteer/
├── CLAUDE.md  # Global context (Tier 0)
├── src/
│   ├── core/  # Engine foundation
│   │   ├── context.md
│   │   ├── components/  # Transform, Renderer, RigidBody, Collider, Velocity
│   │   ├── systems/  # Renderer, PhysicsSystem + PhysicsPhases, Pod drains
│   │   ├── recipe/  # Entity creation system
│   │   └── index.ts
│   ├── character/  # Character controller
│   │   ├── context.md
│   │   ├── components/  # Velocity, CharacterController
│   │   ├── systems/  # Movement system
│   │   ├── factories/  # Entity creation
│   │   └── index.ts
│   ├── camera/  # Camera system
│   │   ├── context.md
│   │   ├── components/  # Camera components
│   │   ├── systems/  # Camera system
│   │   ├── controllers/  # OrbitCamera
│   │   └── index.ts
│   ├── input/  # Input handling
│   │   ├── context.md
│   │   ├── components/  # Input component
│   │   ├── systems/  # Input system
│   │   └── index.ts
│   └── index.ts  # Main exports
├── example/  # Demo application
│   ├── src/
│   │   └── main.ts
│   ├── index.html
│   ├── package.json
│   └── vite.config.ts
├── layers/
│   ├── structure.md  # Project-level context (Tier 1)
│   └── context-template.md  # Template for context files
├── dist/  # Built output
├── package.json
├── tsconfig.json
├── vite.config.ts
├── .prettierrc  # Code formatting config
├── .prettierignore  # Prettier ignore patterns
├── eslint.config.js  # Linting configuration
└── README.md
```

## Architecture

Unity DOTS-inspired with explicit update batches and Pods:

- **SetupBatch**: BeginPod → Input/setup → EndPod
- **FixedBatch**: BeginPod → PhysicsBuild → Character → PhysicsStep → PhysicsSync → EndPod
- **DrawBatch**: BeginPod → Interpolation/Camera/Renderer → EndPod

## Entry points

- Package entry: src/index.ts (exports all features: core, character, camera, input, Game API)
- Vite plugin: src/vite/index.ts (shalloteer() plugin for automatic WASM setup)
- High-level API: Game class and singleton for Roblox-style entity creation
- Example app: example/src/main.ts (demo Three.js application)

## Naming Conventions

**All files and directories use kebab-case**

- Files: `transform.ts`, `rigidbody.ts`, `physics-system.ts`, `pod.ts`
- Directories: `src/`, `core/`, `character/`, `camera/`, `input/`
- Components: kebab-case files containing PascalCase exports (`rigidbody.ts` exports `RigidBody`)
- Systems: kebab-case with `-system` suffix (`physics-system.ts` exports `PhysicsSystem`)
- Multi-word concepts: kebab-case (`orbit-camera.ts`, `character-controller.ts`)

## Configuration

- TypeScript: tsconfig.json (strict mode, ES2020 target, DOM types)
- Build: vite.config.ts (library mode, ESM output, DTS generation)
- Package: package.json (shalloteer engine, dual exports, peer dependencies: bitecs, three, vite-plugin-wasm)
- Code Quality: eslint.config.js (TypeScript linting), .prettierrc (formatting)

## Where to add code

- Core component → src/core/components/[component-name].ts
- Core system → src/core/systems/[system-name].ts
- Recipe/Cook system → src/core/recipe/[file-name].ts
- Character feature → src/character/components|systems|factories/[file-name].ts
- Camera feature → src/camera/components|systems|controllers/[file-name].ts
- Input feature → src/input/components|systems/[file-name].ts
- New feature → src/[feature-name]/[subfolder]/[file-name].ts
- Demo/Example → example/src/[example-file].ts
