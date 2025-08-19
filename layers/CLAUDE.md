# AI Context - Working Agreement

<project-description>
[Short high-level description of the project - MUST be filled during /plant initialization]
</project-description>

<critical-instruction>
**MANDATORY**: Read and analyze [layers/structure.md](layers/structure.md) before proceeding with ANY task. This file contains essential project architecture, technology stack, and command configurations that directly impact how you should approach every task.
</critical-instruction>

## Context Management System

<context-hierarchy>
The Shallot context system uses a four-tier hierarchy to maintain focused, relevant information:

- **Tier 0 — Global Standards**: `CLAUDE.md` (this file)
  - Purpose: System-wide conventions, principles, and behavioral guidelines
  - Loaded: Always, at conversation start
  
- **Tier 1 — Project Architecture**: `layers/structure.md`
  - Purpose: Technology stack, build commands, directory layout, entry points
  - Loaded: Always, immediately after CLAUDE.md
  
- **Tier 2 — Component Context**: `context.md` files in component folders
  - Purpose: Component-specific purpose, scope, dependencies, patterns
  - Loaded: Selectively based on task relevance
  
- **Tier 3 — Implementation**: Source code files
  - Purpose: Actual implementation details
  - Loaded: As needed during work
</context-hierarchy>

## Core Development Principles

<rules>
<rule name="context-priority" severity="critical">
<description>Always load and understand appropriate context before taking any action</description>
<rationale>Working without context leads to inconsistent patterns, duplicated code, and architectural violations</rationale>
<example type="good">
User: "Add user authentication"
Assistant: First, let me check the existing authentication patterns in your project...
[Reads layers/structure.md, searches for auth-related code]
I see you're using JWT with Express middleware. I'll follow this pattern...
</example>
<example type="bad">
User: "Add user authentication"
Assistant: I'll create a new authentication system using Passport.js...
[Proceeds without checking existing patterns]
</example>
</rule>

<rule name="no-history" severity="critical">
<description>Code and context files must NEVER reference their own change history</description>
<rationale>History references cause severe context rot, making files increasingly confusing over time</rationale>
<example type="good">
// user.service.ts
export class UserService {
  async getUser(id: string): Promise<User> {
    return await this.db.users.findById(id);
  }
}
</example>
<example type="bad">
// user.service.ts
export class UserService {
  // Changed from getUserById to getUser for consistency
  // Previously returned UserDto, now returns User entity
  async getUser(id: string): Promise<User> {
    // Was using this.repository, changed to this.db
    return await this.db.users.findById(id);
  }
}
</example>
</rule>

<rule name="simplicity" severity="high">
<description>Prioritize simple, elegant, and readable solutions over complex abstractions</description>
<rationale>Simple code is easier to understand, debug, and modify; premature optimization creates maintenance burden</rationale>
<example type="good">
// Direct, clear implementation
function calculateDiscount(price: number, discountPercent: number): number {
  return price * (1 - discountPercent / 100);
}
</example>
<example type="bad">
// Over-engineered for a simple calculation
class DiscountCalculatorFactory {
  createCalculator(type: DiscountType): IDiscountStrategy {
    return new PercentageDiscountStrategy(new PriceNormalizer());
  }
}
</example>
</rule>

<rule name="single-responsibility" severity="high">
<description>Each file and function should have exactly one clear purpose</description>
<rationale>Single-responsibility components are easier to test, reuse, and understand</rationale>
<example type="good">
// auth.validator.ts - Only validates auth data
export const validateLoginRequest = (data: unknown) => { /* validation logic */ }

// auth.service.ts - Only handles auth business logic
export class AuthService { /* auth operations */ }

// auth.controller.ts - Only handles HTTP routing
export class AuthController { /* route handlers */ }
</example>
<example type="bad">
// user.ts - Does everything
export class User {
  validate() { /* validation */ }
  save() { /* database operations */ }
  sendEmail() { /* email logic */ }
  renderHTML() { /* view logic */ }
}
</example>
</rule>

<rule name="reuse-first" severity="high">
<description>Always search for and reuse existing code before writing new implementations</description>
<rationale>Reduces codebase size, maintains consistency, and leverages tested solutions</rationale>
<implementation>
1. Search for similar functionality using Grep tool
2. Check component context.md files for existing patterns
3. Review dependencies in package.json/requirements.txt
4. Only write new code if no suitable solution exists
</implementation>
</rule>

<rule name="self-documenting-code" severity="medium">
<description>Write code that explains itself through clear naming and structure</description>
<rationale>Good code rarely needs comments; clear names and structure communicate intent better than comments</rationale>
<example type="good">
function isUserEligibleForDiscount(user: User): boolean {
  const hasCompletedFirstPurchase = user.purchaseCount > 0;
  const isWithinPromotionalPeriod = Date.now() < PROMO_END_DATE;
  return hasCompletedFirstPurchase && isWithinPromotionalPeriod;
}
</example>
<example type="bad">
// Check if user can get discount
function check(u: User): boolean {
  // c is count, t is time
  const c = u.p > 0;
  const t = Date.now() < 1735689600000; // Dec 31, 2024
  return c && t; // return true if both conditions met
}
</example>
</rule>

<rule name="single-source-truth" severity="high">
<description>Each piece of data should have exactly one authoritative source</description>
<rationale>Multiple sources of truth lead to synchronization bugs and data inconsistencies</rationale>
<example type="good">
// Single source: database
const user = await db.users.findById(id);
// Derived view: computed from source
const userSummary = computeSummaryFromUser(user);
</example>
<example type="bad">
// Multiple sources that can diverge
let userCache = { id: 1, name: "John" };
let userData = { id: 1, name: "John" };
localStorage.setItem("user", JSON.stringify({ id: 1, name: "John" }));
</example>
</rule>

<rule name="declarative-design" severity="medium">
<description>Favor declarative, data-driven approaches over imperative logic</description>
<rationale>Declarative code is more predictable, testable, and easier to modify</rationale>
<example type="good">
const validationRules = {
  email: { required: true, pattern: /^[^\s@]+@[^\s@]+\.[^\s@]+$/ },
  age: { required: true, min: 18, max: 120 }
};

function validate(data: any, rules: Rules): ValidationResult {
  return Object.entries(rules).map(([field, rule]) => 
    validateField(data[field], rule)
  );
}
</example>
<example type="bad">
function validateUser(user: any): boolean {
  if (!user.email) return false;
  if (!user.email.includes("@")) return false;
  if (!user.age) return false;
  if (user.age < 18) return false;
  if (user.age > 120) return false;
  return true;
}
</example>
</rule>

<rule name="fail-fast" severity="high">
<description>Surface errors immediately and visibly rather than suppressing them</description>
<rationale>Hidden errors create debugging nightmares; visible failures lead to quick fixes</rationale>
<example type="good">
function processPayment(amount: number): void {
  if (amount <= 0) {
    throw new Error(`Invalid payment amount: ${amount}`);
  }
  // Process payment
}
</example>
<example type="bad">
function processPayment(amount: number): void {
  if (amount <= 0) {
    console.log("Warning: invalid amount");
    return; // Silently fails
  }
  // Process payment
}
</example>
</rule>

<rule name="simplicity-over-compatibility" severity="medium">
<description>Prioritize clean, simple design over backwards compatibility unless explicitly required</description>
<rationale>The established design principles make refactoring safe and straightforward when breaking changes are needed</rationale>
<decision-framework>
- Default: Choose the simpler solution
- Only add compatibility layers when explicitly requested
- Document breaking changes clearly in commit messages
</decision-framework>
</rule>
</rules>

## Security Requirements

<security-rules>
<rule name="input-validation">
<description>Validate and sanitize all external inputs</description>
<implementation>
- Validate types, ranges, and formats at system boundaries
- Use established validation libraries (e.g., zod, joi, yup)
- Reject invalid input early with clear error messages
</implementation>
</rule>

<rule name="secrets-management">
<description>Never commit secrets or sensitive data</description>
<implementation>
- Secrets ONLY in environment variables
- Use .env.example files with dummy values
- Never log tokens, passwords, or PII
- Add sensitive files to .gitignore immediately
</implementation>
</rule>

<rule name="authentication">
<description>Implement authentication at appropriate boundaries</description>
<implementation>
- Use gateway/middleware patterns for auth checks
- Validate tokens server-side
- Implement proper session management
- Follow OWASP authentication guidelines
</implementation>
</rule>
</security-rules>

## Tool Usage Optimization

<tool-patterns>
<pattern name="parallel-search">
<description>Execute multiple searches simultaneously for better performance</description>
<example>
// When understanding a feature, search in parallel:
1. Grep for function usage
2. Glob for related files  
3. Read relevant context.md files
All in a single tool invocation set
</example>
</pattern>

<pattern name="context-before-code">
<description>Always read context files before diving into implementation</description>
<sequence>
1. Read layers/structure.md
2. Read relevant context.md files
3. Search for existing patterns
4. Only then read/write code
</sequence>
</pattern>

<pattern name="batch-operations">
<description>Combine related operations in single tool calls</description>
<example>
// Good: Single message with multiple tool calls
[Grep for "authenticate"], [Glob for "*.auth.*"], [Read auth/context.md]

// Bad: Sequential individual calls
[Grep for "authenticate"]
[Wait for result]
[Glob for "*.auth.*"]
[Wait for result]
</example>
</pattern>
</tool-patterns>

## Task Execution Framework

<task-analysis>
When receiving any request:

1. **Context Loading Phase**
   - Load layers/structure.md to understand project architecture
   - Identify relevant components from user request
   - Load appropriate context.md files
   - Search for existing related code

2. **Planning Phase**
   - Break down complex tasks into steps
   - Identify dependencies and impacts
   - Plan parallel operations where possible
   - Consider security implications

3. **Implementation Phase**
   - Follow existing patterns found in context search
   - Implement using established project conventions
   - Maintain single responsibility principle
   - Write self-documenting code

4. **Verification Phase**
   - Run lint/type checking if commands available
   - Verify no duplicate functionality created
   - Ensure security requirements met
   - Confirm context files remain accurate
</task-analysis>

## Proactive Behaviors

<proactive-guidelines>
<behavior name="command-discovery">
<trigger>When build/test/lint commands needed but not found in structure.md</trigger>
<action>
1. Search package.json, Makefile, or similar for commands
2. Ask user for correct commands if not found
3. Suggest updating structure.md with discovered commands
</action>
</behavior>

<behavior name="pattern-detection">
<trigger>When multiple files follow a clear pattern not documented</trigger>
<action>
1. Identify the pattern through code analysis
2. Follow the pattern in new implementations
3. Suggest documenting pattern in relevant context.md
</action>
</behavior>

<behavior name="context-gaps">
<trigger>When working in a folder lacking context.md</trigger>
<action>
1. Analyze folder structure and purpose
2. Create appropriate context.md if substantial component
3. Follow context-template.md format
</action>
</behavior>
</proactive-guidelines>

## Available Tools

<tools>
- **Context7**: Documentation fetching service for external libraries and frameworks
  - Usage: Fetch accurate, up-to-date documentation when needed
  - Invoke via `/context7` command or direct tool use
</tools>
