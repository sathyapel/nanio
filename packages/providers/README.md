# @nanio/providers

Resilient and robust API clients for Gemini, OpenAI, Claude, and xAI (Grok) for the `nanio` agentic framework.

## Key Features

*   **Token Bucket Rate Limiter**: Rate-limiting policies per VIP, PREMIUM, or FREE tier.
*   **Circuit Breaker**: Isolated service protection that trips to OPEN state after consecutive failures.
*   **Exponential Jittered Retries**: Intelligent retry logic for transient network or server errors.
*   **Standardized Clients**:
    *   `GeminiModel` (native token counting & batch embed)
    *   `OpenAIModel`
    *   `ClaudeModel`
    *   `XAIModel` & `XAIImageClient` (Grok text/image capabilities)
