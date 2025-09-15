# 🍃 /nourish

Complete conversation by updating context and applying cleanup.

## Auto-Loaded Context:

@CLAUDE.md
@layers/structure.md
@layers/context-template.md

User arguments: "$ARGUMENTS"

## Steps

### 1. Identify Changes

- Detect modified, added, or deleted files in conversation
- Map changes to their parent folders and components

### 2. Update Context Chain

Traverse upward through context tiers (see CLAUDE.md):

- **Tier 2**: Update relevant `context.md` files to reflect current code state
- **Tier 1**: Update `layers/structure.md` if structure/commands/stack changed
- Follow all rules from CLAUDE.md, especially "No History" principle

### 3. Apply Cleanup

Fix obvious issues encountered:

- Remove comments; code should be self-explanatory without comments
- Remove dead code and unused files
- Consolidate duplicate patterns
- Apply CLAUDE.md principles (simplicity, reuse, single responsibility)

### 4. Verify

- Context accurately reflects current state
- Project is leaner or same size as before
- No history references in code or context

## Output

Report conversation completion: updated context files and improvements made.

## Guidelines

- When updating context, don't over-specify implementation details
- If changes were internal (e.g. business logic), it may not be necessary to update context
- Context should be even shorter after updates, avoiding context rot
