# 🌱 /plant - Project Initialization Command

<command-purpose>
Initialize comprehensive context management structure for a project by analyzing its architecture, patterns, and conventions.
</command-purpose>

## Auto-Loaded Context:
@/CLAUDE.md
@/layers/structure.md

User arguments: "$ARGUMENTS"

## Execution Framework

<phase name="1-project-analysis">
### Step 1: Deep Project Analysis

<analysis-strategy>
**Parallel Discovery Pattern**:
Execute simultaneously for comprehensive understanding:
```
[Glob "**/*.{json,yaml,toml,ini,conf}"]  // Config files
[Glob "**/package.json|**/requirements.txt|**/Gemfile|**/go.mod"]  // Dependencies
[Glob "**/Makefile|**/Dockerfile|**/*.sh"]  // Build/deploy
[Read README.md if exists]  // Project documentation
[LS root directory]  // Top-level structure
```
</analysis-strategy>

<technology-detection>
**Stack Identification Checklist**:
□ **Language & Runtime**
  - Check file extensions (.js/.ts, .py, .java, .go, etc.)
  - Read package managers (package.json, requirements.txt, pom.xml)
  - Identify runtime version from config files

□ **Framework Detection**
  - Web frameworks (Express, Django, Rails, Spring)
  - Frontend frameworks (React, Vue, Angular, Svelte)
  - Testing frameworks (Jest, Pytest, RSpec)
  
□ **Build & Tooling**
  - Build tools (Webpack, Vite, Gradle, Make)
  - Linters (ESLint, Pylint, RuboCop)
  - Formatters (Prettier, Black, gofmt)
  - Type systems (TypeScript, MyPy, Flow)

□ **Infrastructure**
  - Containerization (Docker, Kubernetes)
  - CI/CD (GitHub Actions, Jenkins, CircleCI)
  - Cloud platforms (AWS, GCP, Azure config)
</technology-detection>

<pattern-recognition>
**Code Pattern Analysis**:
1. **Architecture Style**
   - MVC, MVP, MVVM
   - Microservices vs Monolith
   - Event-driven vs Request-response
   - Functional vs Object-oriented

2. **File Organization**
   - Feature-based vs Layer-based
   - Module boundaries
   - Naming conventions
   - Test file locations

3. **Common Patterns**
   - Dependency injection style
   - Error handling approach
   - Validation strategy
   - State management
</pattern-recognition>

<example name="project-analysis-output">
Analyzed Project: E-commerce Platform

**Technology Stack**:
- Runtime: Node.js 18.x with TypeScript 5.0
- Framework: Express.js with REST API
- Database: PostgreSQL with Prisma ORM
- Testing: Jest + Supertest
- Build: Vite, Docker

**Architecture**:
- Pattern: MVC with service layer
- Organization: Feature-based modules
- API: RESTful with /api/v1 prefix
- Auth: JWT with refresh tokens

**Conventions**:
- Naming: camelCase for files, PascalCase for classes
- Validation: Zod schemas
- Error handling: Centralized middleware
- Testing: __tests__ folders per feature
</example>
</phase>

<phase name="2-structure-completion">
### Step 2: Complete Structure Documentation

<structure-template-filling>
**Stack Section**:
```markdown
## Stack
- Runtime/Platform: [Detected runtime] [Version]
- Language: [Primary language] [Version] with [Package manager]
- Framework: [Main framework] [Version]
- Database: [DB type] with [ORM/driver]
- Testing: [Test framework] with [Test runner]
- Tooling: Formatter [Name], Linter [Name], Types [System]
```

**Commands Section**:
Extract from package.json, Makefile, or similar:
```markdown
## Commands
- Dev: [Found dev command]
- Build: [Found build command]
- Test: [Found test command]
- Lint: [Found lint command]
- Format: [Found format command]
- Deploy: [Found deploy command if any]
```

**Layout Section**:
Map actual directory structure:
```markdown
## Layout
[project-root]/
├── src/               # Main source code
│   ├── [feature1]/    # Feature module
│   ├── [feature2]/    # Feature module
│   └── shared/        # Shared utilities
├── tests/             # Test files
├── config/            # Configuration
└── docs/              # Documentation
```

**Entry Points**:
Identify main application entry:
```markdown
## Entry Points
- Application: [main file path]
- API Server: [server file path]
- Worker/Jobs: [worker file path if any]
- CLI: [CLI entry if any]
```
</structure-template-filling>

<example name="filled-structure">
## Stack
- Runtime/Platform: Node.js 18.17.0
- Language: TypeScript 5.0.4 with npm
- Framework: Express.js 4.18.2
- Database: PostgreSQL 15 with Prisma 5.0
- Testing: Jest 29.5 with Supertest
- Tooling: Formatter Prettier, Linter ESLint, Types TypeScript

## Commands
- Dev: npm run dev
- Build: npm run build
- Test: npm test
- Lint: npm run lint
- Format: npm run format
- Type Check: npm run typecheck

## Layout
ecommerce/
├── src/
│   ├── products/      # Product catalog management
│   ├── orders/        # Order processing
│   ├── users/         # User management
│   ├── auth/          # Authentication
│   └── shared/        # Shared utilities
├── tests/             # Integration tests
├── prisma/            # Database schema
└── config/            # App configuration

## Entry Points
- Application: src/index.ts
- API Server: src/server.ts
- Worker: src/workers/queue.ts
- CLI: src/cli/index.ts
</example>
</phase>

<phase name="3-context-creation">
### Step 3: Generate Component Context Files

<context-generation-strategy>
For each major component/folder identified:

1. **Analyze Component**
   - Read main files to understand purpose
   - Identify public interfaces/exports
   - Determine dependencies
   - Note data models used

2. **Generate Context**
   - Create context.md following template
   - Fill purpose from code analysis
   - Map file structure
   - Document entry points
   - List dependencies

3. **Validate Context**
   - Ensure no placeholder text
   - Verify accuracy against code
   - Check completeness
</context-generation-strategy>

<component-priority-matrix>
| Component Type | Priority | Context Depth |
|---------------|----------|---------------|
| Core business logic | High | Detailed with all methods |
| API/Routes | High | Endpoints and middleware |
| Data models | High | Schema and relationships |
| Authentication | High | Security boundaries |
| Utilities | Medium | Public interfaces only |
| Tests | Low | Testing patterns only |
| Config | Low | Key settings only |
</component-priority-matrix>

<example name="generated-context">
Generated: src/products/context.md

```markdown
# Product Management Context

## Purpose
- Product catalog CRUD operations
- Inventory tracking and updates
- Product search and filtering
- Category management
- Price calculations with discounts

## Layout
```
products/
├── context.md           # This file
├── product.model.ts     # Product data model
├── product.service.ts   # Business logic
├── product.controller.ts # HTTP handlers
├── product.routes.ts    # API routes
├── product.validation.ts # Input validation
└── __tests__/          # Component tests
```

## Scope
- In-scope: Product data, inventory, pricing, categories
- Out-of-scope: Order processing, payment, shipping

## Entry Points
- GET /api/v1/products - List products with filters
- GET /api/v1/products/:id - Get single product
- POST /api/v1/products - Create product (admin)
- PUT /api/v1/products/:id - Update product (admin)
- DELETE /api/v1/products/:id - Delete product (admin)

## Dependencies
- Internal: auth (for admin checks), shared/database
- External: prisma (ORM), zod (validation)
```
</example>

<context-creation-rules>
1. **One context.md per significant folder** (not every folder)
2. **Focus on current state** - no history or migration notes
3. **Keep descriptions concise** - link to code for details
4. **Document patterns** not implementation details
5. **Include test approach** if notable
</context-creation-rules>
</phase>

<phase name="4-validation">
### Step 4: Validate and Finalize

<validation-checklist>
□ **CLAUDE.md Completeness**
  - Project description filled
  - All relevant rules included
  - No template placeholders remain

□ **Structure.md Accuracy**
  - Stack matches actual dependencies
  - Commands work when executed
  - Layout reflects real structure
  - Entry points are correct

□ **Context Coverage**
  - All major components have context.md
  - No placeholder text remains
  - Purposes accurately described
  - Dependencies correctly mapped

□ **Consistency Check**
  - Naming conventions consistent
  - All contexts follow template
  - No conflicting information
  - Hierarchy makes sense
</validation-checklist>

<validation-commands>
```bash
# Test that commands work
npm run dev --dry-run
npm run build --dry-run
npm test --listTests

# Verify entry points exist
ls -la src/index.ts
ls -la src/server.ts

# Check for placeholders
grep -r "\[.*\]" --include="*.md" layers/
grep -r "\[.*\]" --include="context.md" .

# Verify context files created
find . -name "context.md" -type f
```
</validation-commands>

<quality-criteria>
**Good Context System**:
- Developer can understand project in 5 minutes
- New features follow existing patterns
- Context guides without overwhelming
- Updates are straightforward

**Warning Signs**:
- Too many context files (over-documentation)
- Vague descriptions ("handles stuff")
- Outdated information
- Missing critical components
</quality-criteria>
</phase>

## Decision Framework

<decision-tree name="folder-context-creation">
```
Is folder a major component/feature?
├─ Yes → Create context.md
│   └─ Does it have subfolders?
│       ├─ Yes → Consider separate contexts
│       └─ No → Single context sufficient
└─ No → Is it shared/utilities?
    ├─ Yes → Create if complex
    └─ No → Skip (config, assets, etc.)
```
</decision-tree>

<decision-tree name="detail-level">
```
Is component business-critical?
├─ Yes → Detailed context with examples
└─ No → Is it frequently modified?
    ├─ Yes → Standard context
    └─ No → Is it complex?
        ├─ Yes → Focus on patterns
        └─ No → Minimal context
```
</decision-tree>

## Output Format

<output-template>
### 🌱 Project Initialized: [Project Name]

**Detected Stack**:
- Language: [Language/Framework]
- Build: [Build system]
- Testing: [Test framework]

**Created Structure**:
- ✅ Updated CLAUDE.md with project description
- ✅ Completed layers/structure.md with [X] commands
- ✅ Created [N] component context files

**Component Contexts**:
- `[component1]/context.md` - [Brief purpose]
- `[component2]/context.md` - [Brief purpose]
- `[component3]/context.md` - [Brief purpose]

**Patterns Identified**:
- [Pattern 1]: [Where used]
- [Pattern 2]: [Where used]

**Next Steps**:
1. Review and customize CLAUDE.md rules
2. Run `/peel` to load context for first task
3. Use `/nourish` after completing work

Ready to begin development with context-aware assistance.
</output-template>

<example name="complete-output">
### 🌱 Project Initialized: E-commerce Platform

**Detected Stack**:
- Language: TypeScript/Express.js
- Build: Vite + Docker
- Testing: Jest + Supertest

**Created Structure**:
- ✅ Updated CLAUDE.md with e-commerce platform description
- ✅ Completed layers/structure.md with 6 commands
- ✅ Created 8 component context files

**Component Contexts**:
- `src/products/context.md` - Product catalog management
- `src/orders/context.md` - Order processing workflow
- `src/users/context.md` - User account management
- `src/auth/context.md` - Authentication and authorization
- `src/payments/context.md` - Payment processing
- `src/inventory/context.md` - Stock management
- `src/api/context.md` - API gateway and routing
- `src/shared/context.md` - Shared utilities and helpers

**Patterns Identified**:
- Validation: Zod schemas in validation.ts files
- Error handling: Centralized middleware pattern
- Testing: __tests__ folders with .spec.ts files
- API: RESTful with /api/v1 versioning

**Next Steps**:
1. Review and customize CLAUDE.md rules
2. Run `/peel` to load context for first task
3. Use `/nourish` after completing work

Ready to begin development with context-aware assistance.
</example>

## Error Recovery

<error-scenarios>
<scenario name="unknown-stack">
**Situation**: Cannot identify technology stack
**Recovery**:
1. Ask user for primary language/framework
2. Create minimal structure.md
3. Fill in details as discovered
4. Suggest user updates structure.md
</scenario>

<scenario name="complex-monorepo">
**Situation**: Multiple projects in monorepo
**Recovery**:
1. Ask which project to initialize
2. Create contexts at appropriate level
3. Note monorepo structure in CLAUDE.md
4. Consider separate contexts per package
</scenario>

<scenario name="existing-contexts">
**Situation**: Some context files already exist
**Recovery**:
1. Preserve existing contexts
2. Update only if outdated
3. Fill gaps with new contexts
4. Report what was preserved vs created
</scenario>
</error-scenarios>

## Performance Optimization

<optimization-patterns>
1. **Parallel Analysis**: Read multiple config files simultaneously
2. **Smart Sampling**: Read key files rather than entire codebase
3. **Pattern Caching**: Remember detected patterns for context generation
4. **Incremental Creation**: Generate contexts as needed, not all at once
5. **Bulk Operations**: Create multiple files in single operation
</optimization-patterns>
