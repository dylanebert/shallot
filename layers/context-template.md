# Component Context Template

<template-purpose>
Structured template for documenting component-level context. Each significant folder should have a context.md file following this format.
</template-purpose>

## Component Metadata

<metadata>
**Component Name**: [Component name]
**Type**: [Feature/Service/Utility/Infrastructure]
**Owner**: [Team or maintainer if applicable]
**Created**: [Date if known]
**Stability**: [Stable/Beta/Experimental]
**Dependencies Count**: [Number of external dependencies]
</metadata>

## Purpose & Responsibilities

<purpose>
**Primary Purpose**:
[Clear, single-sentence description of what this component does]

**Key Responsibilities**:
- [Responsibility 1 - specific capability]
- [Responsibility 2 - specific capability]
- [Responsibility 3 - specific capability]

**Business Value**:
[How this component contributes to the application's goals]
</purpose>

## Component Structure

<structure>
```
[component-folder]/
├── context.md              # This documentation file
├── index.[ext]             # Public API exports
├── [component].[type].[ext] # Core implementation files
├── types.[ext]             # Type definitions
├── utils/                  # Component-specific utilities
├── constants.[ext]         # Component constants
└── __tests__/             # Component tests
    ├── unit/              # Unit tests
    └── integration/       # Integration tests
```

**File Purposes**:
- `index.[ext]`: [What it exports]
- `[main-file].[ext]`: [What it contains]
- `[other-file].[ext]`: [What it contains]
</structure>

## Scope Definition

<scope>
**In Scope**:
- [Specific feature/capability this component handles]
- [Data or operations it owns]
- [Decisions it makes]

**Out of Scope**:
- [What this component explicitly does NOT handle]
- [Responsibilities that belong to other components]
- [Operations it delegates elsewhere]

**Boundary Interactions**:
- Receives: [What data/requests it accepts]
- Provides: [What data/services it offers]
- Events: [What events it emits/consumes]
</scope>

## Public Interface

<interface>
**Exported Functions**:
```typescript
// Main operations this component exposes
functionName(params): ReturnType - Description
functionName2(params): ReturnType - Description
```

**Exported Types**:
```typescript
// Key types/interfaces this component exports
type TypeName = {
  // structure
}
```

**Exported Constants**:
```typescript
// Important constants exposed
const CONSTANT_NAME = value;
```
</interface>

## Entry Points & Routes

<entry-points>
**API Endpoints** (if applicable):
- `GET /path` - [Purpose and response]
- `POST /path` - [Purpose and payload]
- `PUT /path/:id` - [Purpose and params]
- `DELETE /path/:id` - [Purpose]

**Event Handlers** (if applicable):
- `event:name` - [What triggers it and what it does]
- `queue:job` - [Job type and processing]

**CLI Commands** (if applicable):
- `command subcommand` - [What it does]
</entry-points>

## Dependencies

<dependencies>
**Internal Dependencies**:
- `[../other-component]` - [Why needed and what's used]
- `[../../shared/util]` - [Specific utilities used]

**External Dependencies**:
- `[package-name]` - [Purpose and version constraints]
- `[service-name]` - [External service integration]

**Dependency Rules**:
- This component should NOT depend on: [List of components]
- Components that depend on this: [List of dependents]
</dependencies>

## Data & State Management

<data-management>
**Data Models**:
- `[ModelName]`: [What it represents]
- `[ModelName2]`: [What it represents]

**State Management**:
- Stateless/Stateful: [Which and why]
- State Storage: [Where state lives if stateful]
- State Shape: [Structure of state if applicable]

**Caching Strategy**:
- Cache Location: [Where cached data lives]
- Cache Keys: [Pattern for cache keys]
- TTL: [Cache expiration strategy]
</data-management>

## Configuration

<configuration>
**Environment Variables**:
- `ENV_VAR_NAME` - [Purpose, required/optional, default]
- `ANOTHER_VAR` - [Purpose, required/optional, default]

**Configuration Options**:
```javascript
{
  option1: "value", // Description
  option2: 123,     // Description
}
```

**Feature Flags**:
- `FLAG_NAME` - [What it controls]
</configuration>

## Error Handling

<error-handling>
**Error Types**:
- `[ErrorType]` - [When thrown and why]
- `[ValidationError]` - [Validation failure cases]

**Error Recovery**:
- [Error scenario] → [Recovery strategy]
- [Error scenario] → [Recovery strategy]

**Logging**:
- Info: [What informational events are logged]
- Warning: [What warnings are logged]
- Error: [What errors are logged]
</error-handling>

## Testing Approach

<testing>
**Test Strategy**:
- Unit Tests: [What aspects are unit tested]
- Integration Tests: [Integration points tested]
- Test Data: [How test data is managed]

**Key Test Scenarios**:
1. [Critical scenario to test]
2. [Edge case to test]
3. [Error condition to test]

**Mocking Strategy**:
- [What is mocked and how]
- [Test doubles used]
</testing>

## Performance Considerations

<performance>
**Optimization Points**:
- [Optimization implemented or needed]
- [Caching strategy if applicable]
- [Batch processing approach if applicable]

**Resource Usage**:
- Memory: [Expected memory usage]
- CPU: [Processing intensity]
- I/O: [Database/network calls]

**Scalability**:
- Horizontal: [Can scale horizontally? How?]
- Vertical: [Vertical scaling considerations]
- Bottlenecks: [Known limitations]
</performance>

## Security Considerations

<security>
**Security Measures**:
- Authentication: [How auth is handled]
- Authorization: [Permission checks]
- Input Validation: [Validation approach]
- Data Sanitization: [How data is sanitized]

**Sensitive Data**:
- [What sensitive data this component handles]
- [How it's protected]
</security>

## Maintenance Notes

<maintenance>
**Common Issues**:
- [Issue]: [Solution]
- [Issue]: [Solution]

**Monitoring**:
- Metrics: [What metrics to track]
- Alerts: [What conditions trigger alerts]
- Health Checks: [How to verify component health]

**Future Improvements**:
- [ ] [Planned improvement]
- [ ] [Technical debt to address]
- [ ] [Optimization opportunity]
</maintenance>

## Examples

<examples>
**Basic Usage**:
```javascript
// Example of how to use this component
import { mainFunction } from './component';

const result = await mainFunction({
  param1: 'value',
  param2: 123
});
```

**Advanced Usage**:
```javascript
// More complex usage pattern
// [Include relevant example]
```

**Integration Example**:
```javascript
// How this component integrates with others
// [Include integration example]
```
</examples>

---

<template-notes>
**When creating a context.md file**:
1. Remove these template notes
2. Fill in all applicable sections
3. Delete sections that don't apply
4. Keep descriptions concise but complete
5. Focus on current state, not history
6. Link to code for implementation details
7. Update whenever component significantly changes
</template-notes>