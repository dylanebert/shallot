# Project Structure

<project-overview>
Comprehensive map of the project's technology stack, directory layout, and development workflow.
</project-overview>

## Technology Stack

<stack-definition>
**Runtime & Platform**:
- Runtime: [Name] [Version]
- Platform: [Platform if applicable]
- Package Manager: [npm/yarn/pip/gem/cargo]

**Core Technologies**:
- Language: [Primary language] [Version]
- Framework: [Main framework] [Version]
- Database: [Database type] [Version] with [ORM/Driver]
- Cache: [Redis/Memcached if used]

**Development Tools**:
- Build: [Build tool/bundler]
- Test Runner: [Test framework]
- Linter: [Linting tool]
- Formatter: [Code formatter]
- Type Checker: [Type system if applicable]

**Infrastructure**:
- Container: [Docker/Podman if used]
- CI/CD: [GitHub Actions/Jenkins/etc]
- Deployment: [Platform/method]
</stack-definition>

## Development Commands

<command-reference>
| Command | Purpose | Usage |
|---------|---------|--------|
| **Dev** | Start development server | `[command]` |
| **Build** | Build for production | `[command]` |
| **Test** | Run test suite | `[command]` |
| **Lint** | Check code quality | `[command]` |
| **Format** | Auto-format code | `[command]` |
| **Type Check** | Verify types | `[command]` |
| **Deploy** | Deploy to production | `[command]` |

**Command Execution Patterns**:
- Run in parallel: `[command1] & [command2]`
- Run sequentially: `[command1] && [command2]`
- Watch mode: `[command] --watch`
</command-reference>

## Directory Layout

<directory-structure>
```
[project-root]/
├── src/                    # Source code
│   ├── [feature]/         # Feature modules
│   │   ├── context.md     # Component context
│   │   └── ...           # Implementation files
│   ├── shared/            # Shared utilities
│   └── ...               # Other components
├── tests/                 # Test files
├── config/                # Configuration
├── docs/                  # Documentation
├── scripts/               # Build/deploy scripts
└── [build-output]/        # Compiled files
```

**Directory Conventions**:
- **Feature Organization**: [Feature-based/Layer-based]
- **Test Location**: [Alongside code/Separate folder]
- **Config Management**: [Environment files/Config folder]
- **Asset Handling**: [Public folder/CDN/Bundled]
</directory-structure>

## Application Entry Points

<entry-points>
**Primary Entry Points**:
- Main Application: `[path/to/main.ext]`
- Server Entry: `[path/to/server.ext]`
- Client Entry: `[path/to/client.ext]`

**Secondary Entry Points**:
- Worker Processes: `[path/to/workers]`
- CLI Interface: `[path/to/cli]`
- Admin Panel: `[path/to/admin]`
- API Gateway: `[path/to/api]`

**Route Structure**:
- API Routes: `/api/[version]/[resource]`
- Static Assets: `/static/[type]/[file]`
- Health Check: `/health`
- Documentation: `/docs`
</entry-points>

## Configuration Management

<configuration>
**Environment Variables**:
- Development: `.env.development`
- Production: `.env.production`
- Test: `.env.test`
- Example: `.env.example`

**Configuration Files**:
- App Config: `[config/app.js]`
- Database: `[config/database.js]`
- Services: `[config/services.js]`

**Secrets Management**:
- Storage: Environment variables only
- Rotation: [Method if applicable]
- Validation: [Validation approach]
</configuration>

## Code Organization Patterns

<organization-patterns>
**Architecture Pattern**: [MVC/MVP/MVVM/Clean/Hexagonal]

**Module Structure**:
```
[module]/
├── index.[ext]           # Public exports
├── [module].model.[ext]  # Data model
├── [module].service.[ext] # Business logic
├── [module].controller.[ext] # Request handling
├── [module].routes.[ext] # Route definitions
├── [module].validation.[ext] # Input validation
└── __tests__/           # Module tests
```

**Naming Conventions**:
- Files: [camelCase/kebab-case/snake_case]
- Classes: [PascalCase]
- Functions: [camelCase]
- Constants: [UPPER_SNAKE_CASE]
- Interfaces: [IPascalCase/PascalCase]
</organization-patterns>

## Development Workflow

<workflow>
**Branch Strategy**:
- Main Branch: `[main/master]`
- Feature Branches: `feature/[feature-name]`
- Bug Fixes: `fix/[bug-description]`
- Releases: `release/[version]`

**Commit Conventions**:
- Format: `[type]([scope]): [description]`
- Types: feat, fix, docs, style, refactor, test, chore
- Example: `feat(auth): add OAuth2 integration`

**Code Review Process**:
1. Create feature branch
2. Implement changes
3. Run tests locally
4. Create pull request
5. Pass CI checks
6. Code review
7. Merge to main
</workflow>

## Testing Strategy

<testing-approach>
**Test Types**:
- Unit Tests: `[test-runner] [pattern]`
- Integration Tests: `[test-runner] [pattern]`
- E2E Tests: `[test-runner] [pattern]`

**Test File Patterns**:
- Unit: `*.[spec|test].[ext]`
- Integration: `*.integration.[spec|test].[ext]`
- E2E: `*.e2e.[spec|test].[ext]`

**Coverage Requirements**:
- Minimum Coverage: [X]%
- Critical Paths: [Y]%
- New Code: [Z]%
</testing-approach>

## Build & Deployment

<build-deployment>
**Build Process**:
1. Clean previous build
2. Run linter
3. Run type checker
4. Run tests
5. Bundle/compile code
6. Optimize assets
7. Generate sourcemaps

**Deployment Stages**:
- Development: [Auto-deploy on push]
- Staging: [Deploy on PR merge]
- Production: [Manual release]

**Health Checks**:
- Endpoint: `/health`
- Database: Connection pool status
- Dependencies: External service checks
- Memory: Usage thresholds
</build-deployment>

## Decision Matrices

<decision-matrix name="where-to-add-code">
| Code Type | Location | Example |
|-----------|----------|---------|
| New Feature | `src/[feature]/` | User profiles → `src/profiles/` |
| Shared Utility | `src/shared/utils/` | Date formatter → `src/shared/utils/date.js` |
| API Endpoint | `src/api/[version]/` | New resource → `src/api/v1/resource.js` |
| Database Model | `src/models/` | User model → `src/models/user.js` |
| Middleware | `src/middleware/` | Auth check → `src/middleware/auth.js` |
| Configuration | `config/` | API keys → `config/services.js` |
| Test | `[location]/__tests__/` | User tests → `src/user/__tests__/` |
</decision-matrix>

<decision-matrix name="dependency-selection">
| Need | Preferred Solution | Avoid |
|------|-------------------|--------|
| HTTP Client | [axios/fetch] | Multiple HTTP libraries |
| Date Handling | [date-fns/dayjs] | Moment.js (deprecated) |
| Validation | [zod/joi/yup] | Custom validation |
| State Management | [Redux/MobX/Zustand] | Global variables |
| Styling | [CSS Modules/Styled Components] | Inline styles |
| Testing | [Jest/Vitest] | Multiple test runners |
</decision-matrix>

## Common Patterns

<patterns>
**Error Handling Pattern**:
```javascript
try {
  const result = await operation();
  return { success: true, data: result };
} catch (error) {
  logger.error('Operation failed', error);
  throw new AppError(error.message, error.code);
}
```

**Validation Pattern**:
```javascript
const schema = z.object({
  field: z.string().min(1).max(100),
  // ... other fields
});

const validated = schema.parse(input);
```

**API Response Pattern**:
```javascript
{
  success: boolean,
  data?: any,
  error?: {
    code: string,
    message: string,
    details?: any
  },
  meta?: {
    page: number,
    total: number
  }
}
```
</patterns>

## Troubleshooting Guide

<troubleshooting>
**Common Issues**:

| Issue | Cause | Solution |
|-------|-------|----------|
| Build fails | Missing dependencies | Run `[package manager] install` |
| Tests fail | Env variables missing | Copy `.env.example` to `.env.test` |
| Type errors | Outdated types | Run `[type check command]` |
| Lint errors | Format issues | Run `[format command]` |
| Port in use | Dev server running | Kill process on port or change port |

**Debug Commands**:
- Check processes: `ps aux | grep [process]`
- Check ports: `lsof -i :[port]`
- Clear cache: `[package manager] cache clean`
- Reset modules: `rm -rf node_modules && [install]`
</troubleshooting>

## Performance Considerations

<performance>
**Optimization Checklist**:
□ Code splitting implemented
□ Lazy loading for routes
□ Images optimized
□ Bundle size monitored
□ Database queries optimized
□ Caching strategy in place
□ CDN configured
□ Compression enabled

**Performance Targets**:
- Build time: < [X] seconds
- Start time: < [Y] seconds
- Response time: < [Z] ms
- Bundle size: < [N] KB
</performance>
