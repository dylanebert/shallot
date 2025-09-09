# Project Structure

[One-line description of the project - MUST be filled during /plant initialization]

## Stack

- Runtime: [e.g., Node.js, Bun, Deno]
- Language: [e.g., TypeScript, JavaScript, Python]
- Framework: [e.g., React, Vue, Express]
- Build: [e.g., Vite, Webpack, Rollup]
- Database: [e.g., PostgreSQL, MongoDB, SQLite]
- Testing: [e.g., Jest, Vitest, Mocha]

## Commands

- Dev: `[command]` ([description])
- Build: `[command]` ([description])
- Test: `[command]` ([description])
- Lint: `[command]` ([description])
- Format: `[command]` ([description])
- [Other]: `[command]` ([description])

## Layout

```
[project-name]/
├── CLAUDE.md  # Global context (Tier 0)
├── src/  # Main source code
│   ├── [module]/  # Module/feature folder
│   │   ├── context.md  # Module context
│   │   ├── [subfolders]/  # Module structure
│   │   └── index.[ext]  # Module entry
│   └── index.[ext]  # Main entry
├── layers/
│   ├── structure.md  # Project-level context (Tier 1)
│   └── context-template.md  # Template for context files
├── [config-files]  # Configuration files
└── README.md
```

## Architecture

[Describe the high-level architecture pattern]

- **Pattern**: [e.g., MVC, Microservices, Event-driven, ECS]
- **Layers**: [e.g., Presentation → Business → Data]
- **Flow**: [e.g., Request → Controller → Service → Repository]

## Entry points

- Main entry: [path] ([description])
- API entry: [path] ([description])
- CLI entry: [path] ([description])
- [Other]: [path] ([description])

## Naming Conventions

[Describe file and folder naming patterns]

- Files: [pattern and examples]
- Directories: [pattern and examples]
- Classes/Functions: [pattern and examples]
- Variables/Constants: [pattern and examples]

## Configuration

- [Config Type]: [file] ([description])
- [Config Type]: [file] ([description])
- [Config Type]: [file] ([description])

## Where to add code

- [Feature type] → [path pattern]
- [Feature type] → [path pattern]
- [Feature type] → [path pattern]
- New feature → [path pattern]
