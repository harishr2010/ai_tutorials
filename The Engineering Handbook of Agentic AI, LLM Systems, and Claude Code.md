# The Engineering Handbook of Agentic AI, LLM Systems, and Claude Code
### A Production-Grade Reference Manual for AI Engineers & System Architects

---

## Preface & Core Architecture Overview

Modern AI engineering shifts the paradigm from simple single-turn prompt-response interactions to **autonomous agentic state machines**, **hierarchical subagent orchestration**, **deterministic tool protocols**, and **stringent context lifecycle management**.

Building reliable production systems on frontier models (such as the Anthropic Claude 3.5 / 3.7 families) requires treating probabilistic LLMs as specialized compute engines bound by strict software engineering constructs: deterministic interfaces, explicit boundary gating, type validation, and idempotent loop control.

This handbook maps out the 5 core domains evaluated in the CCA-F exam:
1. **Domain 1 (D1): Agentic Architecture & Orchestration Control Flow**
2. **Domain 2 (D2): Tool Engineering, Model Context Protocol (MCP) & Interface Design**
3. **Domain 3 (D3): Claude Code Runtime, Configuration Scoping & Extensibility**
4. **Domain 4 (D4): Prompt Engineering, Evaluation, Structured Extraction & Batching**
5. **Domain 5 (D5): Context Management, Memory Lifecycles & Attention Mechanics**

---

# Domain 1: Agentic Architecture & Orchestration Control Flow

## 1.1 The Core Agentic Loop: State Machines, Termination, and Stop Reasons

At the heart of every autonomous agent is an iterative control loop bridging model completions with programmatic tool executions.

### The Fundamental Flaw of Natural Language & Fixed-Cap Termination
A frequent engineering failure is relying on natural language cues or hard iteration caps to control the execution lifecycle:
* **Natural Language Halting (`"I am done"`, `"Final Answer"`):** Models are probabilistic token predictors. Minor changes in system instructions, temperature, or user context can alter phrasing, causing the orchestrator parser to miss the signal and trigger runaway loops.
* **Fixed Iteration Caps as Primary Stop Conditions:** Imposing a hard cap (e.g., stopping after 5 turns) guarantees that non-trivial multi-step tasks (e.g., investigating a nested dispute or multi-dependency refactoring) fail prematurely (seen in exam questions Q2, Q23, and Q35).

### The Canonical Deterministic State Machine
The correct pattern uses the API's structured protocol: inspect `response.stop_reason`.

```
                    +-----------------------------+
                    |        Start Turn           |
                    | (Append Messages to Context)|
                    +--------------+--------------+
                                   |
                                   v
                    +-----------------------------+
                    |    Call Claude Messages API |
                    +--------------+--------------+
                                   |
                     Inspect `response.stop_reason`
                                   |
            +----------------------+----------------------+
            |                                             |
            v                                             v
    stop_reason == "tool_use"                     stop_reason == "end_turn"
            |                                             |
            v                                             v
+-----------------------------+               +-----------------------------+
| 1. Extract Tool Calls Array |               | Complete Task Execution     |
| 2. Run PreTool Hooks/Gates  |               | Return Assistant Final      |
| 3. Execute Tools (I/O)      |               | Message to User             |
| 4. Append `tool_result`     |               +-----------------------------+
+--------------+--------------+
               |
               +-----> (Loop back to Next Turn)
```

### Reference Implementation: Canonical Agentic Loop

```python
import os
from typing import List, Dict, Any, Callable
from anthropic import Anthropic
from anthropic.types import Message, ContentBlock

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def execute_agentic_loop(
    initial_prompt: str,
    system_prompt: str,
    tools: List[Dict[str, Any]],
    tool_dispatcher: Callable[[str, Dict[str, Any]], Dict[str, Any]],
    safety_max_iterations: int = 25  # Safety fallback only, NOT primary control logic
) -> str:
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": initial_prompt}
    ]
    
    iteration = 0
    while iteration < safety_max_iterations:
        iteration += 1
        
        response: Message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            tools=tools,
            tool_choice={"type": "auto"}
        )
        
        # 1. State: The agent decides to emit tool calls
        if response.stop_reason == "tool_use":
            # Append assistant message (containing text blocks and tool_use blocks)
            messages.append({"role": "assistant", "content": response.content})
            
            tool_results_blocks = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_use_id = block.id
                    
                    try:
                        # Execute the tool implementation
                        result_data = tool_dispatcher(tool_name, tool_input)
                        tool_results_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": str(result_data),
                            "is_error": False
                        })
                    except Exception as err:
                        tool_results_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": f'{{"error": "{str(err)}"}}',
                            "is_error": True
                        })
            
            # Append user message containing all tool execution results
            messages.append({"role": "user", "content": tool_results_blocks})
            continue

        # 2. State: Task complete
        elif response.stop_reason == "end_turn":
            # Extract final text
            text_blocks = [b.text for b in response.content if hasattr(b, "text")]
            return "\n".join(text_blocks)

        elif response.stop_reason == "max_tokens":
            raise RuntimeError("Agent exceeded output token limit before completing turn.")
            
        else:
            raise ValueError(f"Unhandled stop_reason: {response.stop_reason}")

    raise TimeoutError(f"Safety circuit-breaker triggered after {safety_max_iterations} iterations.")
```

---

## 1.2 Multi-Agent Orchestration Patterns: Coordinators, Subagents, and Context Isolation

Large, ambiguous problems overwhelm a single context window. The standard pattern is a **Hierarchical Coordinator-Subagent Architecture**.

```
                           +----------------------+
                           |  Research / Task     |
                           |  Coordinator Agent   |
                           +----------+-----------+
                                      |
                 +--------------------+--------------------+
                 | (Dispatches via `Task` tool calls)      |
                 v                                         v
     +-----------------------+                 +-----------------------+
     |  Web Search Subagent  |                 | Document Subagent     |
     |  Context: Isolated    |                 | Context: Isolated     |
     |  Tools: [web_search]  |                 | Tools: [read_doc]     |
     +-----------+-----------+                 +-----------+-----------+
                 |                                         |
                 +--------------------+--------------------+
                                      |
                          Aggregates Output Results
                                      v
                           +----------------------+
                           |  Synthesis Subagent  |
                           |  Context: Isolated   |
                           |  Input: Explicit Map |
                           +----------------------+
```

### The Inviolable Axiom: Context Isolation
Subagents do **not** inherit context, memory, conversation history, or tools from their parent coordinator, nor do they share state with peer subagents (Q13, Q50, Q57). 
* **The "Context Amnesia" Bug:** If a coordinator invokes a `generate_code` subagent and subsequently invokes a `write_tests` subagent, the test-generation subagent has zero awareness of the generated code unless the coordinator explicitly includes the code or file artifacts in the test subagent's invocation payload (Q50).
* **The "Synthesis Blindness" Bug:** If the coordinator runs independent search and document analysis agents, the final synthesis subagent will completely omit web results if the coordinator only feeds it the document output (Q13).

### Spawning Subagents Programmatically
To give a coordinator the authority to spawn child subagents, two conditions must be met:
1. The tool definition for subagent delegation (e.g., named `Task` or `spawn_subagent`) must be explicitly present in the coordinator's `allowedTools` or declared tool configuration (Q1).
2. The subagent's execution environment must be initialized with its own dedicated system prompt, allowed tools, and isolated context buffer.

### Parallel Execution vs Sequential Serialization
When subagent operations are orthogonal (e.g., searching web literature while parsing internal balance sheets), sequential execution doubles latency (Q9).
* **Mechanism:** The coordinator emits **multiple tool call blocks** in a single API turn response.
* **SDK Orchestrator Action:** The runner executes all tool calls concurrently via asynchronous tasks (`asyncio.gather` or thread workers), constructs the matching array of `tool_result` blocks, and returns them in a single batch back to the coordinator.

```python
# Emitting parallel subagent tasks in Python
import asyncio

async def handle_parallel_tool_calls(tool_blocks: list, registry: dict):
    tasks = []
    for block in tool_blocks:
        tool_fn = registry[block.name]
        tasks.append(tool_fn(**block.input))
    
    # Run all subagents concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return [
        {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": str(res)
        }
        for block, res in zip(tool_blocks, results)
    ]
```

---

## 1.3 Deterministic Safety: Hooks, PreToolUse Policy Gates, and Sequences

Probabilistic instructions in system prompts cannot guarantee compliance on safety-critical, legal, or financial operations (Q18, Q21, Q26).

| Approach | Reliability | Failure Mode | When to Use |
| :--- | :--- | :--- | :--- |
| **System Prompt Rule** | $85\% - 95\%$ | Overridden by complex reasoning, edge cases, or jailbreaks | Low-stakes stylistic preferences |
| **Few-Shot Examples** | $90\% - 97\%$ | Degrades on out-of-distribution inputs | Guiding complex edge-case taxonomy |
| **Architectural / Hook Gates** | **$100\%$** | Fails only on code bugs / configuration errors | Financial limits, RBAC, PII, destructive actions |

### Structural Sequence Enforcement (Prerequisite Gating)
Suppose an organization requires that `verify_identity` must run before `process_refund`, or `get_order_history` must execute before calculating refunds (Q18, Q26).
* **Wrong Pattern:** A `PostToolUse` hook on `get_order_history`. If the model decides to bypass `get_order_history` and directly invokes `process_refund`, the post-hook never executes.
* **Right Pattern:** A **`PreToolUse` hook on `process_refund`**. It verifies the session state ledger to confirm that an identity verification token or order history payload exists. If absent, the call is blocked programmatically and an error is returned to the model.

```python
class AgentSessionState:
    def __init__(self):
        self.verified_identity: bool = False
        self.order_history_retrieved: bool = False

def pre_tool_use_enforcer(tool_name: str, tool_args: dict, state: AgentSessionState):
    if tool_name == "process_refund":
        amount = tool_args.get("amount", 0.0)
        
        # Policy Gate 1: Amount Cap (Q21)
        if amount > 500.0:
            raise PermissionError(
                f"Action blocked: Refund amount ${amount} exceeds automated threshold of $500. "
                "Must call escalate_to_human."
            )
            
        # Policy Gate 2: Sequence Enforcement (Q18, Q26)
        if not state.verified_identity:
            raise PermissionError(
                "Action blocked: Tool 'verify_identity' must be executed successfully "
                "prior to calling 'process_refund'."
            )
            
        if not state.order_history_retrieved:
            raise PermissionError(
                "Action blocked: Tool 'get_order_history' must be executed prior to calling 'process_refund'."
            )
```

---

# Domain 2: Tool Engineering, Model Context Protocol (MCP) & Interface Design

## 2.1 Tool Description Engineering: The Selection Signal

LLMs select tools using the semantic meaning of their names, descriptions, and JSON schema field definitions (Q5, Q22, Q42).
* When descriptions are brief, identical, or overlapping (e.g., `analyze_content` vs `analyze_document`), routing collapses to near-random distribution (Q5).
* Tool descriptions must document:
  1. **Primary Purpose:** What specific capability the tool unlocks.
  2. **Accepted Input Schemas:** Data formats (e.g., integer order IDs vs UUID strings).
  3. **Explicit Boundary Constraints:** What the tool **cannot** do, and which tool to use instead.
  4. **Side Effects & Behavioral Consequences:** Explaining what happens if optional fields (like `line_number`) are omitted (Q42).

```json
{
  "name": "lookup_order",
  "description": "Retrieves comprehensive e-commerce order details including line items, shipping status, and billing ledger. Use this tool ONLY when querying specific transaction IDs (format: ORD-XXXXX) or customer purchase histories. Do NOT use this tool for customer profile lookups (use lookup_customer instead). Omitting the 'include_returns' flag defaults to false and excludes historical refund records.",
  "parameters": {
    "type": "object",
    "properties": {
      "order_id": {
        "type": "string",
        "pattern": "^ORD-[0-9]{5}$",
        "description": "The unique 5-digit order identifier, prefixed with 'ORD-'"
      },
      "include_returns": {
        "type": "boolean",
        "description": "Set to true to pull historical return events. Defaults to false."
      }
    },
    "required": ["order_id"]
  }
}
```

---

## 2.2 Structured Error Handling & MCP Protocol Design

When tools fail, returning unstructured raw error strings (such as `"multiple_matches"`, `"Cannot process"`, or raw HTTP 500 traces) causes models to fall into infinite retry loops or guess parameters (Q3, Q27, Q30).

### The Canonical MCP Error Response Contract
Tools must return structured error objects containing classification tags, retry indicators, and remediation metadata:

```json
{
  "isError": true,
  "errorCategory": "transient | business | ambiguous_input | validation | permission",
  "isRetryable": false,
  "message": "Human and model readable summary of the specific issue.",
  "recoveryMetadata": {
    "action": "disambiguate_with_user",
    "candidates": [
      {"id": "C-101", "label": "Jane Doe (jane.d@example.com)"},
      {"id": "C-102", "label": "Jane Doe (j.doe@work.org)"}
    ]
  }
}
```

### Categorization Matrix

| Category | `isRetryable` | System Response / Agent Strategy | Example |
| :--- | :--- | :--- | :--- |
| **`transient`** | `true` | Agent pauses (exponential backoff) or queries fallback replica. | Database timeout, network blip, rate limit. |
| **`business`** | `false` | Agent immediately halts attempts and explains policy to user. | 30-day return window expired, invalid discount. |
| **`ambiguous_input`**| `false` | Agent prompts the user to select from candidate matches. | Query returned multiple customer records. |
| **`validation`** | `false` | Agent reformulates parameters to conform to schema. | Invalid timestamp format, malformed UUID. |

---

## 2.3 Least Privilege & Interface Hardening

General-purpose tools present attack surfaces and lead to capability drift (Q11).
* **The Vulnerability:** Giving a document parsing agent a generic `fetch_url` tool allows it to bypass designated search subagents and execute raw HTTP searches, web crawling, or SSRF attacks (Q11).
* **Hardened Design:** Constrain capability at the schema interface. Replace `fetch_url` with `load_document`, strictly typed to accept specific document identifiers or pre-vetted S3 storage keys:

```json
{
  "name": "load_document",
  "description": "Retrieves the parsed text of an enterprise document from internal object storage. Rejects external web URLs, query engines, and search domains.",
  "parameters": {
    "type": "object",
    "properties": {
      "document_id": {
        "type": "string",
        "pattern": "^doc_[a-zA-Z0-9]{12}$",
        "description": "The internal document storage identifier (e.g., 'doc_ab92x0194kmz')."
      }
    },
    "required": ["document_id"]
  }
}
```

---

## 2.4 MCP Architecture: Resources vs Tools

In the Model Context Protocol (MCP), developers often confuse **Tools** with **Resources** (Q12).

| Capability | Model Interaction | Cost & Latency | Best Used For |
| :--- | :--- | :--- | :--- |
| **MCP Tools** | Dynamic Execution: The model decides to run a tool, waits for execution, receives output in a later turn. | High (Adds intermediate turns, tool dispatch overhead, and latency). | Dynamic actions, updates, API calls, side effects. |
| **MCP Resources**| Declarative Data: Exposed as contextual resources (URIs) loaded up front into the model's awareness. | Low (Zero round-trip discovery turns; read once into context). | Slow-changing corpora, catalog schemas, source directories. |

If an agent runs 3 exploratory tool calls on every turn just to list catalogs or schemas, convert those catalogs to **MCP Resources** to surface them instantly at session initialization (Q12).

---

# Domain 3: Claude Code Runtime, Configuration Scoping & Extensibility

Claude Code is Anthropic's agentic command-line interface and coding engine. Mastering it requires understanding its file system hierarchy, configuration scoping, and CLI operational modes.

## 3.1 The Configuration Inheritance Hierarchy

Configurations load in a strict cascading order. Subdirectory configurations extend and override parents without modifying root templates (Q4, Q58).

```
                      ~/.claude/CLAUDE.md
                      (User / Machine Level)
                      Personal styling, global CLI defaults
                      NEVER checked into Git (Q58)
                                |
                                v
                   <repo-root>/CLAUDE.md
                   (Project / Repository Level)
                   Universal team standards, shared build commands,
                   architecture maps. Checked into Version Control.
                                |
                                v
               <repo-root>/src/services/billing/CLAUDE.md
               (Directory Sub-Tree Scoped Level)
               Targeted overrides: APA citation rules, PCI compliance,
               specific test harness configs (Q4).
```

### Modular Instruction Composition: `@import`
Monolithic 700-line `CLAUDE.md` files degrade token economics and cause instruction interference. Use `@import` to modularize standards across separate files (Q34):

```markdown
<!-- root CLAUDE.md -->
# Engineering Standards Hub

@import ./docs/standards/git-workflow.md
@import ./docs/standards/security-rules.md
@import ./docs/standards/review-guidelines.md
```

---

## 3.2 Conditional Rules: `.claude/rules/` with Path Globs

Placing heterogeneous standards (React frontend, Python backend, Terraform HCL) into a global `CLAUDE.md` leads to **convention bleed**—such as applying Terraform linting checks to React JSX files (Q37, Q49).

To bind conventions strictly to matching file types across any directory, use `.claude/rules/` with YAML frontmatter:

```markdown
<!-- .claude/rules/react-conventions.md -->
---
paths:
  - "src/**/*.tsx"
  - "src/**/*.jsx"
---
# React Component Rules
- Always use functional components with explicit TypeScript interfaces.
- Use Tailwind CSS utility classes; avoid inline styles.
- Co-locate component unit tests as `[Component].test.tsx`.
```

```markdown
<!-- .claude/rules/unit-testing.md -->
---
paths:
  - "**/*.test.ts"
  - "**/*.test.tsx"
  - "tests/**/*.py"
---
# Test Harness Standards (Q49)
- Mock all network boundaries using MSW.
- Do not assert on implementation details; assert on observable DOM / API contracts.
```

---

## 3.3 Custom Skills Engine: `.claude/skills/`

Skills are encapsulated slash-commands that developers and automated pipelines invoke to run standardized workflows (Q41, Q60).

### Skill Anatomy: `SKILL.md`
Skills must be configured with real frontmatter properties rather than prose instructions:

```markdown
<!-- .claude/skills/review/SKILL.md -->
---
description: Run automated deep code review on a directory or file.
argument-hint: "<path/to/target>"
allowed-tools:
  - "Read"
  - "Grep"
  - "PostReviewComment"
context: fork
---

# Code Review Procedure
Review the target path specified in `$1`.
1. Check for SQL injection, unescaped templates, and insecure deserialization.
2. Verify test coverage for all modified business branches.
3. Post inline review comments via `PostReviewComment`.
Do NOT modify files on disk. Do NOT run arbitrary bash commands.
```

### Frontmatter Directives Demystified

* **`context: fork` (Q60):** Spawns an isolated subagent sandbox to execute the skill. The massive output from exploratory file dumps, coverage calculations, and lint runs stays confined to the subagent; only the synthesized result returns to the main conversation.
* **`allowed-tools` (Q41):** Strictly whitelists the tools the skill is authorized to run, programmatically preventing unauthorized file edits or bash executions.
* **`argument-hint` (Q41):** Displays required parameters in the CLI autocomplete dropdown, preventing missing-argument errors.

---

## 3.4 Operational CLI Modes: CI/CD & Enterprise Workflows

Running Claude Code within CI/CD pipelines (GitHub Actions, GitLab CI) requires non-interactive execution (Q32, Q43):

```bash
# Correct non-interactive CI invocation
claude -p "Review this pull request diff against main" \
  --output-format json \
  --json-schema ./schemas/pr-review-schema.json \
  > review-findings.json
```

* **`-p` / `--print` (Q32):** Runs Claude Code non-interactively. Processes the prompt, streams output to stdout, and exits with a standard process exit code (`0` on success). Plausible fake flags like `--headless`, `--batch`, or `--no-input` will fail.
* **`--output-format json` with `--json-schema` (Q43):** Directs the CLI engine to produce structured, parseable JSON conforming to a schema, avoiding fragile regex parsing of free-form terminal prose.
* **Plan Mode vs Direct Execution (Q36):** For multi-layer architectural modifications, entering Plan Mode explores files, builds an AST/import dependency graph, and gets consensus on the plan before touching disk.

---

# Domain 4: Prompt Engineering, Evaluation, Structured Extraction & Batching

## 4.1 Schema Enforcement: Tool Use vs Prose vs JSON Output

When structured output is mandatory for downstream pipelines (such as generating PR review annotations or JSDoc comments), relying on system prompt prose instructions (`"Return JSON"`) or few-shot examples will inevitably produce malformed outputs on out-of-distribution inputs (Q17, Q39, Q52).

### The Hierarchy of Schema Reliability

```
[Most Reliable]   JSON Schema via Native Tool Use (`tool_choice`)
                  Model is constrained to emit JSON conformant to the JSON Schema.
        |
        v         API-Level Structured Output (`response_format` / `--json-schema`)
                  Enforces machine parseable structure.
        |
        v         Few-Shot Prompt Anchoring
                  Probabilistic: demonstrates patterns, but vulnerable to edge cases.
        |
[Least Reliable]  System Prompt Prose Instructions ("You must always output JSON...")
                  Subject to format drift, markdown wrapping, and omission.
```

### Schema Calibration: Handling Nullables and Enums (Q39)
A common schema flaw is making fields strictly required when the underlying source cannot support them:
* **The "Fabrication Trap":** If `suggested_fix` is marked non-nullable and required, the model will invent code fixes for problems that have no remediation (Q39).
* **The Solution:** Use explicit enums that include all valid states, mark fields `nullable: true` when absence is valid, and use prompts to define null conditions.

```json
{
  "type": "object",
  "properties": {
    "file_path": { "type": "string" },
    "line_number": { "type": "integer" },
    "severity": {
      "type": "string",
      "enum": ["critical", "high", "medium", "low", "warning"]
    },
    "suggested_fix": {
      "type": ["string", "null"],
      "description": "Provide the exact code replacement. Pass null ONLY when no clear programmatic remediation exists."
    }
  },
  "required": ["file_path", "line_number", "severity", "suggested_fix"]
}
```

---

## 4.2 Calibration of Evaluator Agents & False-Positive Elimination

LLM-as-a-judge pipelines, automated PR reviewers, and security analyzers suffer from **noise fatigue** if poorly calibrated (Q44, Q53).

### Why Vague Modifiers Fail
Telling an evaluator model to *"be conservative"*, *"only report high-confidence issues"*, or *"report only critical bugs"* fails because "important" is an uncalibrated subjective judgment (Q53).

### Concrete Categorical Filtering (Q53)
Evaluator prompts must provide explicit, mutually exclusive **Report** vs **Ignore** taxonomies:

```markdown
### Review Triage Specification
You must report findings that match the REPORT criteria below. You are strictly forbidden from reporting items in the SKIP category.

#### REPORT Criteria:
- Unhandled null pointer exceptions or type invariant breaches.
- O(N^2) or higher computational loops occurring on incoming request threads.
- Exposure of secrets, API keys, or raw PII in output streams.

#### SKIP Criteria:
- Variable naming style and casing conventions.
- Ordering of imports or alphabetical sort patterns.
- Speculative architectural refactoring ("Consider splitting this into two classes").
```

### Iterative Triage of High-Noise Categories (Q44)
When a review pipeline flags 90%+ false positives, adding few-shot examples often fails to address the root issue.
* **The Engineering Protocol:** Analyze false positive logs, identify the specific category causing the noise (e.g., "ReDoS in safe regex"), **disable that specific category entirely** to restore developer trust, iterate on that category's detection criteria in isolation, and re-enable it only after achieving precision benchmarks (Q44).

---

## 4.3 Workload Distribution: Message Batches API vs Synchronous API

The Anthropic Message Batches API offers a **50% cost reduction** on input and output tokens, but introduces architectural constraints (Q31, Q40).

```
+---------------------------------------------------------------------------------------+
|                                API SELECTION MATRIX                                   |
+---------------------------------------------------------------------------------------+
| Workload Requirement            | Target Protocol   | Architecture Justification      |
+---------------------------------+-------------------+---------------------------------+
| Multi-turn Agentic Loop         | Synchronous API   | Batches API cannot intercept a  |
| (Model calls tool -> tool runs  | (Interactive)     | tool call mid-request, run the  |
| -> model resumes turn) (Q31)    |                   | I/O, and return results.        |
+---------------------------------+-------------------+---------------------------------+
| Developer-Blocking Workflow     | Synchronous API   | Batches have NO latency SLA and |
| (Pre-merge CI gate, live chat)  | (Interactive)     | may take up to 24 hours (Q40).  |
+---------------------------------+-------------------+---------------------------------+
| Latency-Tolerant Auditing       | Message Batches   | Overnight jobs tolerate delays; |
| (Nightly security scans,        | API (Batch)       | save 50% cost. Correlate using  |
| weekly compliance summaries)    |                   | `custom_id` (Q40).              |
+---------------------------------------------------------------------------------------+
```

---

# Domain 5: Context Management, Memory Lifecycles & Attention Mechanics

## 5.1 Attention Mechanics & The Lost-in-the-Middle Phenomenon

Frontier models support 200k+ token windows, but token capacity does **not** equal uniform attention (Q33).

```
   High Attention | \                                                 / | High Attention
  (Primacy Effect)|  \                                               /  | (Recency Effect)
                  |   \                                             /   |
                  |    \                                           /    |
                  |     \                                         /     |
   Low Attention  |      \_______________________________________/      |
  (Attention Dip) |                                                     |
                  +-----------------------------------------------------+
                  0%                 Context Position                 100%
                  (Top / System)         (Middle)                 (End / User)
```

### Manifestation in Production (Q33)
When an agent reviews 18 changed files fed sequentially into one large prompt, files 7–12 fall into the attention dip. They receive superficial analysis and fewer flagged bugs, despite no tokens being truncated.

### Mitigation Strategies
1. **Per-File Decomposed Passes (Q38):** Run independent single-file passes for local correctness, followed by a light integration pass for cross-file contracts.
2. **Structural Anchoring:** Re-inject critical schemas, instructions, and target invariants at the end of the prompt immediately preceding the user prompt trigger.

---

## 5.2 Context Degradation, Progressive Summarization & Structured Scratchpads

In prolonged sessions (e.g., 20+ turns or post-`/compact` executions), relying on standard LLM summarization leads to **semantic drift and constraint amnesia** (Q19, Q46).

### The Failure of Narrative Summarization
When an LLM summarizes a conversation history:
* Specific figures, transaction IDs, and exact timestamps (`$89.99 double-charge on March 14, Order #55201`) degrade into vague narrative summaries (`"Customer discussed a previous billing issue"`).
* Negative architectural constraints (e.g., *"Never make synchronous calls in event handlers"*) are often omitted during progressive summarization, causing the model to generate non-compliant patterns later in the session (Q46).

### The Solution: The Persistent Structured State Ledger
Extract critical business facts out of the rolling conversation history into a structured **Case Facts Block** (or system scratchpad) pinned to the top of subsequent prompts (Q19, Q46):

```markdown
<!-- Case Facts Block: Maintained outside the summarization engine -->
<case_facts_ledger>
  <customer_id>C-4421</customer_id>
  <order_id>#55201</order_id>
  <disputed_charge date="2026-03-14" amount="89.99" currency="USD" type="duplicate" />
  <verified_eligibility>true</verified_eligibility>
  <failed_attempts>1</failed_attempts>
  <invariants>
    <rule>Do not allow event handlers to execute synchronous network calls.</rule>
  </invariants>
</case_facts_ledger>
```

---

## 5.3 Phased Pipeline Execution vs Single-Context Pollution

A common failure mode in code refactoring agents is running **Discovery** and **Implementation** in the same context session (Q56, Q59).

```
[WRONG: Monolithic Context Contamination]
+-----------------------------------------------------------------------------------+
| System Prompt -> Read File 1..47 (Massive Dumps) -> Attempt Implementation        |
| Result: Context is flooded with 150k tokens of noise; model forgets target API.   |
+-----------------------------------------------------------------------------------+

[CORRECT: Phased Architecture with Forked Contexts]
+----------------------------+
| Phase 1: Explore Subagent  |
| Read 47 files, run Grep.   |
| Runs in `context: fork`    |
+-------------+--------------+
              |
              v Extracts only: Interfaces, AST Map, Target Call Sites
+----------------------------+
| Phase 2: Implementation    |
| Receives ONLY the compact  |
| brief. Zero noise. Clean.  |
+----------------------------+
```

* **The Explore Subagent Pattern (Q59):** Searching 47 files across a 200,000-line codebase produces hundreds of thousands of intermediate tokens. Running this in a separate subagent isolates this noise, allowing it to return a concise, structured interface brief to the primary session.
* **Phase-Gate Briefs (Q56):** When transitioning from research/exploration to code generation, compress the phase findings into a clean summary and spin up a fresh context.

---

## 5.4 Self-Review Bias and Context Isolation

When a model is asked to generate code or tests and then evaluate its own output in the same session, review efficacy drops significantly (Q45).
* **The Cognitive Bias:** In the original generation session, the model's context is saturated with its own chain-of-thought rationale, trade-off justifications, and implicit assumptions. When asked to critique its work, it tends to validate its own assumptions rather than look for edge-case coverage gaps (Q45).
* **The Architecture:** Use an **independent Claude instance** initialized with only the target specification and the generated code. With zero exposure to the original generation context, the reviewer approaches the code with no prior bias.

---

# Consolidated System Blueprint: Production Research & Synthesis Engine

The following end-to-end Python implementation demonstrates these concepts working together in a production-style architecture:
* Hierarchical task decomposition with input validation.
* Parallel subagent dispatch via asyncio.
* Structured error categorization and recovery.
* Independent subagent context isolation.
* Dedicated validation and synthesis passes.

```python
"""
Production Multi-Agent Research System
Demonstrating Context Isolation, Structured Tool Responses,
Parallel Execution, and Phased Synthesis.
"""

import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from anthropic import AsyncAnthropic

client = AsyncAnthropic()

# ---------------------------------------------------------
# 1. Structured Schemas (Domain 2 & Domain 4)
# ---------------------------------------------------------

class ToolErrorResponse(BaseModel):
    isError: bool = True
    errorCategory: str = Field(..., description="transient | business | ambiguous_input | validation")
    isRetryable: bool
    message: str
    recoveryMetadata: Optional[Dict[str, Any]] = None

class ResearchFinding(BaseModel):
    source_url: str
    claim: str
    confidence_score: float

class SynthesisOutput(BaseModel):
    report_markdown: str
    supported_claims: List[ResearchFinding]
    coverage_gaps: List[str]

# ---------------------------------------------------------
# 2. Hardened Tool Dispatcher (Domain 2)
# ---------------------------------------------------------

async def mock_academic_search(query: str) -> Dict[str, Any]:
    # Demonstrating transient timeout recovery contract (Q3)
    if "timeout_trigger" in query:
        return ToolErrorResponse(
            errorCategory="transient",
            isRetryable=True,
            message="Database read timed out after 30 seconds."
        ).model_dump()
        
    return {
        "isError": False,
        "results": [
            {"source_url": "https://arxiv.org/abs/2401.9999", "claim": "Post-quantum lattice signatures scale linearly on edge MCU nodes."}
        ]
    }

async def mock_document_reader(document_id: str) -> Dict[str, Any]:
    # Least-privilege constrained tool (Q11)
    if not document_id.startswith("doc_"):
        return ToolErrorResponse(
            errorCategory="validation",
            isRetryable=False,
            message="Invalid document identifier. Tool only accepts doc_ prefix."
        ).model_dump()
        
    return {
        "isError": False,
        "content": "Enterprise IoT devices require under 64KB memory overhead for PQC handshakes."
    }

# ---------------------------------------------------------
# 3. Subagent Execution Worker: Isolated Contexts (Domain 1 & Domain 5)
# ---------------------------------------------------------

async def run_subagent_task(role: str, task_prompt: str, tools: list) -> Dict[str, Any]:
    """
    Subagent running in its own isolated context buffer.
    Never inherits parent session history (Q13, Q50).
    """
    messages = [{"role": "user", "content": task_prompt}]
    
    # Run subagent completion
    response = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2048,
        system=f"You are a specialized {role}. Return your findings strictly as structured facts with claim-to-source mappings.",
        messages=messages
    )
    
    return {
        "role": role,
        "output": response.content[0].text
    }

# ---------------------------------------------------------
# 4. Orchestrator: Coordinator & Dynamic Aggregation (Domain 1)
# ---------------------------------------------------------

async def run_research_pipeline(research_topic: str) -> SynthesisOutput:
    # Phase 1: Planning and Decomposition (Q7, Q14)
    # The coordinator breaks down broad tasks into focused subtasks
    subtasks = [
        {"role": "Academic Researcher", "prompt": f"Extract formal cryptographic proofs on: {research_topic}"},
        {"role": "Industry Systems Analyst", "prompt": f"Extract memory footprint metrics on: {research_topic}"}
    ]
    
    # Phase 2: Parallel Subagent Execution (Q9)
    # Dispatching subagents concurrently to minimize end-to-end latency
    tasks = [
        run_subagent_task(task["role"], task["prompt"], tools=[])
        for task in subtasks
    ]
    subagent_results = await asyncio.gather(*tasks)
    
    # Phase 3: Aggregation with Structured Claim-Source Provenance (Q10, Q13)
    # Aggregating all findings into the synthesis context explicitly
    aggregated_findings_context = "\n\n".join([
        f"### Findings from [{res['role']}]:\n{res['output']}"
        for res in subagent_results
    ])
    
    # Phase 4: Independent Synthesis with Coverage Annotations (Q8, Q45)
    # Using an independent instance to review findings and identify coverage gaps
    synthesis_messages = [
        {
            "role": "user",
            "content": (
                f"Topic: {research_topic}\n\n"
                f"Aggregated Inputs:\n{aggregated_findings_context}\n\n"
                "Synthesize an executive report. Explicitly annotate any subtopics with missing evidence "
                "in a 'coverage_gaps' block. Do not hallucinate or use parametric data for unretrieved sources."
            )
        }
    ]
    
    final_response = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        system="You are a research synthesis agent. Maintain source provenance on every claim.",
        messages=synthesis_messages
    )
    
    return SynthesisOutput(
        report_markdown=final_response.content[0].text,
        supported_claims=[],
        coverage_gaps=["Battery thermal dissipation models under continuous PQC handshakes were unavailable."]
    )

if __name__ == "__main__":
    result = asyncio.run(run_research_pipeline("Post-Quantum Cryptography in Edge IoT Microcontrollers"))
    print("Pipeline Execution Completed.")
    print(result.report_markdown[:300] + "...")
```

---

## Exam Domain Cross-Reference

Use this index to link specific questions from your test review directly to the underlying architecture sections:

* **Domain 1 (Agentic Architecture):**
  * Loop Termination Signals: **Section 1.1** *(Questions: Q2, Q23, Q35)*
  * Subagent Permissions & Allowed Tools: **Section 1.2** *(Question: Q1)*
  * Parallel Tool Calling & Concurrency: **Section 1.2** *(Question: Q9)*
  * Context Isolation & Parameter Passing: **Section 1.2** *(Questions: Q13, Q50, Q51, Q57)*
  * Task Decomposition & Refinement Loops: **Section 1.2** *(Questions: Q7, Q14, Q20, Q48)*
  * Hook Gating & Prerequisite Sequences: **Section 1.3** *(Questions: Q18, Q21, Q26)*
  * Sequential Prompt Chains: **Section 1.3** *(Question: Q29)*
  * Third-Party Tool Output Normalization: **Section 1.3** *(Question: Q25)*

* **Domain 2 (Tool Design & MCP):**
  * Tool Descriptions as Primary Signal: **Section 2.1** *(Questions: Q5, Q22, Q42)*
  * Structured MCP Errors & Retry Semantics: **Section 2.2** *(Questions: Q3, Q27, Q30)*
  * Principle of Least Privilege: **Section 2.3** *(Questions: Q6, Q11)*
  * MCP Resources vs Dynamic MCP Tools: **Section 2.4** *(Question: Q12)*
  * Project MCP Configuration & Token Expansion: **Section 2.4** *(Question: Q54)*

* **Domain 3 (Claude Code Runtime):**
  * Configuration Hierarchy & Scoping: **Section 3.1** *(Questions: Q4, Q15, Q24, Q58)*
  * Modular Rules & `@import`: **Section 3.1** *(Question: Q34)*
  * Conditional Path Scoping via Glob Rules: **Section 3.2** *(Questions: Q37, Q49)*
  * Skill Frontmatter (`context: fork`, `allowed-tools`): **Section 3.3** *(Questions: Q41, Q60)*
  * CI/CD Execution Flags (`-p`, `--output-format`): **Section 3.4** *(Questions: Q32, Q43)*
  * Plan Mode vs Direct Execution: **Section 3.4** *(Question: Q36)*

* **Domain 4 (Prompt Engineering & Extraction):**
  * Schema Enforcement (Tool Use vs Prose): **Section 4.1** *(Questions: Q17, Q39, Q52)*
  * Input-Output Example Calibration: **Section 4.1** *(Question: Q47)*
  * Evaluator Calibration & False-Positive Elimination: **Section 4.2** *(Questions: Q44, Q53)*
  * Interview Pattern for Ambiguous Domains: **Section 4.2** *(Question: Q55)*
  * Batch API vs Synchronous API Trade-offs: **Section 4.3** *(Questions: Q31, Q40)*

* **Domain 5 (Context Management & Memory):**
  * The Lost-in-the-Middle Effect: **Section 5.1** *(Question: Q33)*
  * Structured Case-Facts vs Narrative Summaries: **Section 5.2** *(Questions: Q19, Q46)*
  * Claim-to-Source Attribution Preservation: **Section 5.2** *(Questions: Q8, Q10)*
  * Phased Pipelines & Subagent Context Protection: **Section 5.3** *(Questions: Q56, Q59)*
  * Self-Review Bias & Context Isolation: **Section 5.4** *(Question: Q45)*