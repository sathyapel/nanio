# @nanio/observability

Structured JSON observability, request context propagation, PII scrubbing, and real-time execution cost-budget calculations for the `nanio` agentic framework.

## Key Features

*   **`AsyncLocalStorage` Context**: Standardized request metadata context (trace IDs, user IDs, cost accumulators) propagating down async callback stacks automatically.
*   **Structured JSON Logger**: Custom logger injecting request context on every single trace line.
*   **PII & Secret Scrubbing**: Automated regex-based scrubbing of emails, phone numbers, API keys, and access tokens before outputting logs.
*   **Pluggable Cost Tracker**: Real-time USD pricing calculators and budget warnings with backends for Memory, Redis, MongoDB, and PostgreSQL (`PgCostStore`).
*   **Performance Metrics & Violations**: Metric latency histograms and violations with Postgres (`PgPerfStore`), Redis, and Memory stores.
