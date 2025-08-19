# 🧄 /peel - Context Loading Command

<command-purpose>
Load precisely targeted context for the current conversation based on task requirements.
</command-purpose>

## Auto-Loaded Context:
@/CLAUDE.md
@/layers/structure.md

User arguments: "$ARGUMENTS"

## Execution Framework

<phase name="1-parse-requirements">
### Step 1: Parse and Analyze Work Requirements

<analysis-checklist>
□ Parse user arguments/request for key terms and concepts
□ Identify primary task type (feature, bug, refactor, research)
□ Determine affected system components
□ Assess scope (single file, component, cross-component, system-wide)
□ Identify potential dependencies
</analysis-checklist>

<task-classification>
Classify the task to determine context depth:

**Narrow Tasks** (single component):
- Bug fixes in specific files
- Adding methods to existing classes
- Updating single component logic
→ Load: Component context.md only

**Medium Tasks** (2-3 components):
- Feature additions spanning modules
- Refactoring related components
- API endpoint implementations
→ Load: Primary and dependent component contexts

**Broad Tasks** (system-wide):
- Architecture changes
- New major features
- Performance optimizations
→ Load: All relevant component contexts

**Research Tasks** (exploration):
- Understanding existing implementation
- Finding usage patterns
- Debugging complex issues
→ Load: Minimal context, rely on search tools
</task-classification>

<example name="parsing-user-request">
User: "/peel implement user profile editing with image upload"

Analysis Output:
- Task Type: Feature addition
- Primary Components: user, profile, media/upload
- Scope: Cross-component (3 components)
- Dependencies: auth (for user context), storage (for images)
- Context Needed: user/context.md, profile/context.md, media/context.md
</example>
</phase>

<phase name="2-load-context">
### Step 2: Load Targeted Context

<loading-strategy>
**Parallel Loading Pattern** (for efficiency):
```
// Execute simultaneously in single tool call:
[Read user/context.md]
[Read profile/context.md] 
[Read media/context.md]
[Glob "**/*profile*"]
[Grep "uploadImage|imageUpload"]
```

**Hierarchical Loading** (for complex features):
1. Load parent component context
2. Identify subcomponents from parent
3. Load relevant subcomponent contexts
4. Skip unrelated sibling components

**Lazy Loading** (for research tasks):
1. Load minimal context initially
2. Use search tools to explore
3. Load additional context as patterns emerge
</loading-strategy>

<context-relevance-matrix>
| User Request Contains | Load These Contexts | Skip These |
|----------------------|---------------------|-------------|
| "auth", "login", "user" | auth/, user/, session/ | admin/, analytics/ |
| "api", "endpoint" | api/, routes/, middleware/ | frontend/, styles/ |
| "database", "model" | models/, db/, migrations/ | views/, public/ |
| "ui", "component" | components/, views/, styles/ | backend/, db/ |
| "test", "spec" | tests/, __tests__/, spec/ | docs/, scripts/ |
</context-relevance-matrix>

<optimization-rules>
1. **Never load everything** - Context window pollution reduces performance
2. **Defer code reading** - Load context.md first, code only when needed
3. **Use search over reading** - Grep/Glob for specific patterns vs reading entire files
4. **Cache understanding** - Note key findings to avoid re-reading
</optimization-rules>
</phase>

<phase name="3-confirm-understanding">
### Step 3: Confirm Scope and Understanding

<confirmation-template>
Based on the request "$ARGUMENTS", I've loaded context for:

**Primary Focus**: [component/feature name]
- Loaded: [list of context.md files loaded]
- Purpose: [brief statement of what will be done]

**Scope Boundaries**:
- In scope: [what will be modified/added]
- Out of scope: [what won't be touched]
- Dependencies: [external components that may be affected]

**Key Patterns Identified**:
- [Pattern 1]: [where it's used]
- [Pattern 2]: [where it's used]

**Assumptions**:
- [Any assumptions made about requirements]
- [Technology choices based on existing patterns]

Ready to proceed with [specific action].
</confirmation-template>

<example name="scope-confirmation">
Based on the request "implement user profile editing with image upload", I've loaded context for:

**Primary Focus**: User profile management with media handling
- Loaded: user/context.md, profile/context.md, media/context.md
- Purpose: Add profile editing capabilities with image upload support

**Scope Boundaries**:
- In scope: Profile edit form, image upload endpoint, storage integration
- Out of scope: User registration, authentication flow, image processing
- Dependencies: Auth middleware (for user validation), Storage service (for images)

**Key Patterns Identified**:
- Form validation: Using zod schemas (found in user/validation.ts)
- File uploads: Multer middleware pattern (found in media/upload.ts)
- API structure: RESTful routes with /api/v1 prefix

**Assumptions**:
- Following existing validation patterns with zod
- Using established multer configuration for uploads
- Maintaining current API versioning scheme

Ready to proceed with implementing the profile editing feature.
</example>
</phase>

## Decision Trees

<decision-tree name="context-loading-depth">
```
Is task clearly scoped to single file?
├─ Yes → Load only that file's parent context.md
└─ No → Is task a bug fix?
    ├─ Yes → Load context for affected component + error context
    └─ No → Is task a new feature?
        ├─ Yes → Load contexts for all touched components
        └─ No → Is task research/exploration?
            ├─ Yes → Minimal context, rely on search
            └─ No → Load broad context for safety
```
</decision-tree>

<decision-tree name="search-vs-read">
```
Need to understand code pattern?
├─ Pattern name known? 
│   ├─ Yes → Use Grep with specific pattern
│   └─ No → Use Glob for file discovery
└─ Need full file content?
    ├─ File path known?
    │   ├─ Yes → Read specific file
    │   └─ No → Glob first, then read
    └─ Multiple files needed?
        ├─ Yes → Batch read in parallel
        └─ No → Single read operation
```
</decision-tree>

## Error Recovery

<error-handling>
<scenario name="missing-context-files">
**Situation**: No context.md found in expected location
**Recovery**:
1. Check parent directory for context
2. Search for README or docs in component
3. Analyze code structure directly
4. Create context.md if component is substantial
</scenario>

<scenario name="ambiguous-scope">
**Situation**: User request could apply to multiple components
**Recovery**:
1. Load contexts for all potential components
2. Present options to user for clarification
3. Proceed with most likely interpretation
4. Note assumptions clearly
</scenario>

<scenario name="conflicting-patterns">
**Situation**: Different components use different patterns
**Recovery**:
1. Identify which pattern is newer/preferred
2. Follow pattern of primary component being modified
3. Note inconsistency for future cleanup
4. Suggest standardization if significant
</scenario>
</error-handling>

## Output Format

<output-structure>
1. **Brief acknowledgment** of request understanding
2. **Loaded contexts** list (component names only, not full paths)
3. **Identified patterns** that will be followed
4. **Ready statement** with specific next action

Keep output concise - aim for 3-5 lines maximum unless complexity requires more.
</output-structure>

## Performance Guidelines

<performance-rules>
1. **Parallel over sequential**: Load multiple contexts simultaneously
2. **Search over read**: Use Grep/Glob before reading entire files
3. **Lazy over eager**: Load only what's needed initially
4. **Cache over repeat**: Remember findings to avoid re-reading
5. **Defer over preload**: Don't read implementation until needed
</performance-rules>

## Command Examples

<example name="narrow-task">
/peel "fix validation error in user registration"

Output:
Loaded user registration context. Found existing zod validation pattern in auth/validation.ts.
Ready to fix the validation error.
</example>

<example name="medium-task">
/peel "add comment system to blog posts"

Output:
Loaded contexts for blog, comments, and user components. 
Identified existing post-author relationship pattern and moderation queue structure.
Ready to implement commenting following your event-driven pattern.
</example>

<example name="broad-task">
/peel "migrate from REST to GraphQL"

Output:
Loaded API, routing, and data layer contexts across 7 components.
Found existing REST patterns: Express routes, controller/service separation, JSON responses.
Identified GraphQL candidates: User, Post, Comment, Media entities.
Ready to plan migration strategy starting with schema definition.
</example>
