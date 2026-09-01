import os
import importlib
from typing import Optional

EXTERNAL_PROVIDERS = {"gemini", "groq", "mistral", "deepseek"}


def _is_local_mode_enabled() -> bool:
    """Check if LOCAL_MODE is enabled in environment."""
    return os.getenv("LOCAL_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_provider() -> str:
    """Resolve the LLM provider from environment variables.
    
    Priority:
    1. If LOCAL_MODE=1, use 'ollama'
    2. Otherwise, use LLM_PROVIDER env var (if valid external provider)
    3. Fallback to default 'gemini'
    """
    if _is_local_mode_enabled():
        return "ollama"
    
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider in EXTERNAL_PROVIDERS:
        return provider
    return "gemini"


def _validate_api_key(provider: str, api_key_env: str) -> str:
    """Validate that an API key is available for a given provider.
    
    Args:
        provider: Provider name (for error messages)
        api_key_env: Environment variable name to check
    
    Returns:
        The API key value
    
    Raises:
        ValueError: If API key is not set
    """
    api_key = os.getenv(api_key_env, "").strip()
    if not api_key:
        raise ValueError(
            f"API key not set for {provider}. Please set {api_key_env} in your .env file."
        )
    return api_key


def build_chat_llm(temperature: float = 0.0, model: Optional[str] = None, provider: Optional[str] = None):
    """Build a chat LLM instance based on the configured provider.
    
    Supported modes:
    - LOCAL_MODE=1: Uses Ollama (local)
    - External: gemini, groq, mistral, deepseek
    
    Args:
        temperature: Model temperature parameter (0.0 - 1.0)
        model: Optional model name override. If not provided, uses provider-specific defaults.
        provider: Optional provider override. If not specified, resolves from environment (LOCAL_MODE or LLM_PROVIDER).
                 If LOCAL_MODE is enabled and an external provider is requested, raises ValueError.
    
    Returns:
        A LangChain chat model instance
    
    Raises:
        ImportError: If required package is not installed
        ValueError: If provider is not supported, API key is missing, or external provider requested in LOCAL_MODE
    """
    resolved_provider = provider or _resolve_provider()
    
    # If in LOCAL_MODE, don't allow external providers
    if _is_local_mode_enabled() and provider and provider != "ollama":
        raise ValueError(
            f"Cannot use external provider '{provider}' when LOCAL_MODE is enabled. "
            "Set LOCAL_MODE to empty/false to use external providers."
        )

    if resolved_provider == "ollama":
        try:
            chat_ollama_module = importlib.import_module("langchain_ollama")
            ChatOllama = getattr(chat_ollama_module, "ChatOllama")
        except ImportError as exc:
            raise ImportError(
                "`langchain-ollama` is required for Ollama mode. "
                "Install it with: pip install langchain-ollama"
            ) from exc

        model_name = model or os.getenv("OLLAMA_MODEL", "mistral")
        base_url = os.getenv("OLLAMA_BASE_URL", "").strip()

        kwargs = {
            "model": model_name,
            "temperature": temperature,
        }
        if base_url:
            kwargs["base_url"] = base_url

        return ChatOllama(**kwargs)

    if resolved_provider == "gemini":
        # Gemini se sirve a través de OpenRouter (https://openrouter.ai),
        # usando la API compatible con OpenAI en lugar de Google AI Studio.
        try:
            openai_module = importlib.import_module("langchain_openai")
            ChatOpenAI = getattr(openai_module, "ChatOpenAI")
        except ImportError as exc:
            raise ImportError(
                "`langchain-openai` is required for Gemini via OpenRouter mode. "
                "Install it with: pip install langchain-openai"
            ) from exc

        _validate_api_key("Gemini (OpenRouter)", "OPENROUTER_API_KEY")
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()

        model_name = model or os.getenv("GEMINI_MODEL", "google/gemini-2.5-flash-lite")
        # OpenRouter espera ids tipo "<proveedor>/<modelo>"; si el modelo no
        # lleva prefijo (p.ej. "gemini-2.5-flash-lite"), se le antepone "google/".
        if "/" not in model_name:
            model_name = f"google/{model_name}"

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
        )

    if resolved_provider == "groq":
        try:
            groq_module = importlib.import_module("langchain_groq")
            ChatGroq = getattr(groq_module, "ChatGroq")
        except ImportError as exc:
            raise ImportError(
                "`langchain-groq` is required for Groq mode. "
                "Install it with: pip install langchain-groq"
            ) from exc

        _validate_api_key("Groq", "GROQ_API_KEY")
        model_name = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        return ChatGroq(model=model_name, temperature=temperature)

    if resolved_provider == "mistral":
        try:
            mistral_module = importlib.import_module("langchain_mistralai")
            ChatMistralAI = getattr(mistral_module, "ChatMistralAI")
        except ImportError as exc:
            raise ImportError(
                "`langchain-mistralai` is required for Mistral mode. "
                "Install it with: pip install langchain-mistralai"
            ) from exc

        _validate_api_key("Mistral", "MISTRAL_API_KEY")
        model_name = model or os.getenv("MISTRAL_MODEL", "mistral-small-2603")
        return ChatMistralAI(model=model_name, temperature=temperature)

    if resolved_provider == "deepseek":
        # DeepSeek se sirve a través de OpenRouter (https://openrouter.ai),
        # usando la API compatible con OpenAI en lugar de la API nativa de DeepSeek.
        try:
            openai_module = importlib.import_module("langchain_openai")
            ChatOpenAI = getattr(openai_module, "ChatOpenAI")
        except ImportError as exc:
            raise ImportError(
                "`langchain-openai` is required for DeepSeek via OpenRouter mode. "
                "Install it with: pip install langchain-openai"
            ) from exc

        _validate_api_key("DeepSeek (OpenRouter)", "OPENROUTER_API_KEY")
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()

        model_name = model or os.getenv("DEEPSEEK_MODEL", "~deepseek/deepseek-v4-flash-latest")

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER='{resolved_provider}'. Supported external providers: {', '.join(sorted(EXTERNAL_PROVIDERS))}"
    )
