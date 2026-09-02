import asyncio
import time


def invoke_structured_with_retry(chain, inputs: dict, max_retries: int = 3, node_name: str = "unknown"):
    """
    Invoca una chain que usa with_structured_output, reintentando ante fallos
    de validacion/parseo (ValidationError, OutputParserException, errores de red, etc.).
    Devuelve el resultado parseado, o None si todos los intentos fallan.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            print(f"[{node_name}] Attempt {attempt}/{max_retries} failed: {e}", flush=True)
            if attempt < max_retries:
                time.sleep(1.5 * attempt)
    return None


async def invoke_structured_with_retry_async(chain, inputs: dict, max_retries: int = 3, node_name: str = "unknown"):
    """
    Variante async de invoke_structured_with_retry.
    Usa await chain.ainvoke(inputs) y await asyncio.sleep(...) entre reintentos.
    Devuelve el resultado parseado, o None si todos los intentos fallan.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return await chain.ainvoke(inputs)
        except Exception as e:
            print(f"[{node_name}] Attempt {attempt}/{max_retries} failed: {e}", flush=True)
            if attempt < max_retries:
                await asyncio.sleep(1.5 * attempt)
    return None