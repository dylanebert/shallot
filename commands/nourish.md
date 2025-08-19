# 🍃 /nourish - Context Update & Cleanup Command

<command-purpose>
Complete the current conversation by updating context files to reflect changes and applying systematic cleanup to maintain codebase health.
</command-purpose>

## Auto-Loaded Context:
@/CLAUDE.md
@/layers/structure.md

User arguments: "$ARGUMENTS"

## Execution Framework

<phase name="1-change-detection">
### Step 1: Comprehensive Change Detection

<change-analysis>
**File Change Detection**:
```
// Parallel execution for efficiency
[Git diff --name-status]  // Modified files
[Git status --porcelain]   // Untracked files
[Review conversation log]  // Discussed but unchanged files
```

**Change Classification**:
- **Created**: New files added to the project
- **Modified**: Existing files with changes
- **Deleted**: Files removed from project
- **Moved**: Files relocated or renamed
- **Discussed**: Files analyzed but not changed

**Impact Mapping**:
For each changed file, determine:
1. Parent component (folder containing the file)
2. Dependent components (imports from/to this file)
3. Context tier affected (component, project, global)
4. Documentation impact (context.md updates needed)
</change-analysis>

<example name="change-detection-output">
Detected Changes:
- Created: src/user/profile-edit.ts, src/api/upload.ts
- Modified: src/user/routes.ts, src/user/validation.ts
- Discussed: src/auth/middleware.ts (examined for patterns)

Impact Analysis:
- Component 'user': Added profile editing feature
- Component 'api': New upload endpoint added
- Dependencies: 'auth' component provides validation middleware
- Context Updates Needed: user/context.md, api/context.md
</example>

<change-tracking-matrix>
| File Pattern | Context Update Required | Update Type |
|-------------|------------------------|-------------|
| New feature files | Yes - component context | Add to entrypoints, scope |
| Modified business logic | Yes - component context | Update behavior description |
| New API endpoints | Yes - API context + structure.md | Document endpoint |
| Config changes | Yes - structure.md | Update stack/commands |
| Style/formatting | No | Skip |
| Test files | Sometimes - if new test pattern | Note testing approach |
</change-tracking-matrix>
</phase>

<phase name="2-context-updates">
### Step 2: Systematic Context Chain Updates

<update-strategy>
**Bottom-Up Traversal**:
Start from implementation (Tier 3) and work up to global (Tier 0)

1. **Component Level** (context.md files):
   - Update based on file changes
   - Maintain "no history" rule
   - Reflect current state only

2. **Project Level** (structure.md):
   - Update if structure changed
   - Update if new commands added
   - Update if stack modified

3. **Global Level** (CLAUDE.md):
   - Rarely needs updates
   - Only if new patterns established
   - Only if new tools added
</update-strategy>

<context-update-rules>
<rule name="no-history-enforcement">
**CRITICAL**: Never write "was X, now Y" or "changed from A to B"
- Wrong: "Changed authentication from JWT to OAuth"
- Right: "Uses OAuth for authentication"
</rule>

<rule name="current-state-only">
Write context as if describing the system for the first time
- Wrong: "Added profile editing to user component"
- Right: "Handles user profiles including viewing and editing"
</rule>

<rule name="preserve-structure">
Maintain existing context.md structure while updating content
- Keep section headers
- Update bullet points
- Preserve formatting style
</rule>
</context-update-rules>

<example name="context-update-before-after">
BEFORE (user/context.md):
```markdown
## Purpose
- User authentication and session management
- Password reset functionality

## Entrypoints
- login(): Authenticates user credentials
- logout(): Terminates user session
```

AFTER (user/context.md):
```markdown
## Purpose
- User authentication and session management
- Password reset functionality
- Profile management including editing and image uploads

## Entrypoints
- login(): Authenticates user credentials
- logout(): Terminates user session
- editProfile(): Updates user profile data
- uploadAvatar(): Handles profile image uploads
```
</example>

<structure-update-triggers>
Update structure.md when:
1. **New build/dev commands** discovered or added
2. **Directory structure** significantly changed
3. **New dependencies** added to package.json/requirements.txt
4. **Entry points** modified or added
5. **Configuration files** created or modified
</structure-update-triggers>
</phase>

<phase name="3-cleanup-operations">
### Step 3: Apply Systematic Cleanup

<cleanup-checklist>
□ **Dead Code Removal**
  - Unused imports
  - Commented-out code blocks
  - Unused variables/functions
  - Empty files

□ **Duplication Consolidation**
  - Repeated logic patterns
  - Similar utility functions
  - Duplicate type definitions
  - Redundant validation rules

□ **Pattern Standardization**
  - Inconsistent naming conventions
  - Mixed async patterns (callbacks vs promises vs async/await)
  - Varying error handling approaches
  - Different validation strategies

□ **File Organization**
  - Files in wrong directories
  - Overly large files needing splitting
  - Misplaced business logic
  - Test files not following convention

□ **Dependency Optimization**
  - Unused npm packages
  - Redundant dependencies
  - Version inconsistencies
  - Missing dependencies
</cleanup-checklist>

<cleanup-examples>
<example name="dead-code-removal">
Found and removed:
- 3 unused imports in user.service.ts
- Commented-out legacy authentication code (lines 45-89)
- Unused validateEmail function replaced by zod schema
</example>

<example name="pattern-consolidation">
Standardized error handling:
- Consolidated 4 different error response formats into single ErrorResponse type
- Unified try-catch patterns across all controllers
- Created shared error middleware
</example>

<example name="file-reorganization">
Restructured for clarity:
- Split 500-line user.controller.ts into separate route files
- Moved validation schemas to dedicated validation/ folder
- Grouped related utilities into shared/utils/
</example>
</cleanup-examples>

<cleanup-philosophy>
**Principle**: Leave code cleaner than you found it
**Balance**: Cleanup related to current work, not wholesale refactoring
**Focus**: Reduce complexity, improve maintainability
**Constraint**: Don't break existing functionality
</cleanup-philosophy>
</phase>

<phase name="4-verification">
### Step 4: Verify Context Accuracy

<verification-checklist>
□ **Context Completeness**
  - All changed components have updated context.md
  - New features documented in appropriate tier
  - Dependencies accurately reflected

□ **No History References**
  - Scan all updated files for "was", "changed", "previously"
  - Ensure clean current-state descriptions
  - Remove any migration or update notes

□ **Code Quality**
  - Run linter if available
  - Run type checker if available
  - Verify no broken imports
  - Ensure tests still pass (if applicable)

□ **File Size**
  - Project same size or smaller (not bloated)
  - No unnecessary files added
  - Cleanup actually removed code
</verification-checklist>

<verification-commands>
```bash
# Verify no history references
grep -r "previously\|was\|changed from" --include="*.md"

# Check for unused dependencies
npm ls --depth=0 (for Node.js)
pip list (for Python)

# Verify no broken imports
npm run typecheck (if available)
python -m py_compile **/*.py (for Python)

# Check file organization
find . -type f -size +500k (large files)
find . -type f -name "*.tmp" -o -name "*.bak" (temp files)
```
</verification-commands>
</phase>

## Decision Framework

<decision-tree name="context-update-depth">
```
Did files change in this component?
├─ Yes → Update component context.md
│   └─ Did structure change significantly?
│       ├─ Yes → Update structure.md
│       └─ No → Component context only
└─ No → Did we discuss this component?
    ├─ Yes → Review context for accuracy
    └─ No → Skip this component
```
</decision-tree>

<decision-tree name="cleanup-aggressiveness">
```
Is cleanup target related to current work?
├─ Yes → Apply cleanup
│   └─ Will cleanup break other components?
│       ├─ Yes → Document needed changes, don't apply
│       └─ No → Apply cleanup
└─ No → Is it a critical issue (security/bug)?
    ├─ Yes → Apply cleanup anyway
    └─ No → Note for future cleanup
```
</decision-tree>

## Output Format

<output-template>
### Conversation Complete: [Task Summary]

**Updated Context Files**:
- `component1/context.md`: [what was updated]
- `component2/context.md`: [what was updated]
- `layers/structure.md`: [if updated, what changed]

**Cleanup Applied**:
- [Cleanup action 1]
- [Cleanup action 2]
- [Cleanup action 3]

**Code Quality**:
- Files affected: [count]
- Lines removed: [count]
- Patterns unified: [list]

**Ready for**: `/clear` to start fresh conversation
</output-template>

<example name="complete-output">
### Conversation Complete: Added User Profile Editing

**Updated Context Files**:
- `user/context.md`: Added profile editing endpoints and image upload scope
- `api/context.md`: Documented new upload endpoint and multer configuration
- `layers/structure.md`: Added image processing commands to build section

**Cleanup Applied**:
- Removed 3 unused validation functions (replaced by zod schemas)
- Consolidated duplicate error handling into shared middleware
- Deleted commented-out legacy authentication code (120 lines)

**Code Quality**:
- Files affected: 8
- Lines removed: 145
- Patterns unified: validation, error handling

**Ready for**: `/clear` to start fresh conversation
</example>

## Error Recovery

<error-scenarios>
<scenario name="context-conflict">
**Situation**: Multiple context files have conflicting information
**Resolution**:
1. Code is source of truth
2. Update all contexts to match code reality
3. Note inconsistencies in output
</scenario>

<scenario name="broken-cleanup">
**Situation**: Cleanup broke existing functionality
**Resolution**:
1. Immediately revert breaking change
2. Document why cleanup failed
3. Create TODO for proper refactoring
</scenario>

<scenario name="missing-commands">
**Situation**: Lint/test commands not found
**Resolution**:
1. Check package.json/Makefile for commands
2. Skip automated verification
3. Suggest adding commands to structure.md
</scenario>
</error-scenarios>

## Performance Optimization

<optimization-patterns>
1. **Batch Updates**: Update multiple context files in single operation
2. **Parallel Cleanup**: Run linter, type checker, tests simultaneously
3. **Incremental Changes**: Update only changed sections of context
4. **Smart Detection**: Use git diff to focus only on changed files
5. **Cached Patterns**: Remember patterns found during conversation
</optimization-patterns>
