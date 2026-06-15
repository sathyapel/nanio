# @nanio/registry

Dynamic model provider routers and fallback registry mechanisms for the `nanio` agentic framework.

## Key Features

*   **`LLMRegistry`**: Model-tier fallback controller that automatically switches models when encountering rate limits or failures.
*   **`FallbackRouter`**: Configurable routing chains supporting tier limits and overrides.
*   **Budget Throttle Status**: Downgrades model routing level dynamically when `CostTracker` reports budget violations.
