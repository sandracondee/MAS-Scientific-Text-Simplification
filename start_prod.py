import os
import signal
import subprocess
import sys
import time
import urllib.request

MCP_SERVERS = [
    {
        "name": "metrics-server",
        "module": "src.mcp.metrics_server",
        "host": os.getenv("MCP_METRICS_HOST", "127.0.0.1"),
        "port": int(os.getenv("MCP_METRICS_PORT", "8020")),
        "path_env": "MCP_METRICS_SERVER_URL",
    },
    {
        "name": "search-server",
        "module": "src.mcp.search_server",
        "host": os.getenv("MCP_SEARCH_HOST", "127.0.0.1"),
        "port": int(os.getenv("MCP_SEARCH_PORT", "8021")),
        "path_env": "MCP_SEARCH_SERVER_URL",
    },
]

READY_TIMEOUT = 120
READY_INTERVAL = 2.0

_procs = []


def _default_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/mcp/"


def _check_ready(host: str, port: int, url: str) -> bool:
    """Return True when the MCP server is listening (any HTTP response).

    FastMCP responde con 3xx/4xx sin las cabeceras del protocolo MCP, así que
    basta con que el servidor devuelva cualquier respuesta HTTP para
    considerarlo "listo". Solo un fallo de conexión indica que aún no escucha.
    """
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


def _wait_ready(config: dict) -> None:
    host = config["host"]
    port = config["port"]
    url = os.getenv(config["path_env"], "") or _default_url(host, port)

    deadline = time.time() + READY_TIMEOUT
    while time.time() < deadline:
        if _check_ready(host, port, url):
            print(f"[start_prod] {config['name']} is ready ({url})", flush=True)
            return
        time.sleep(READY_INTERVAL)

    print(
        f"[start_prod] WARNING: {config['name']} not ready after "
        f"{READY_TIMEOUT}s at {url}. Continuing anyway.",
        flush=True,
    )


def _terminate_children() -> None:
    for proc in _procs:
        if proc.poll() is None:
            print(f"[start_prod] terminating {proc.pid}", flush=True)
            try:
                proc.terminate()
            except Exception:
                pass
    time.sleep(1)
    for proc in _procs:
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
    _procs.clear()


def _signal_handler(signum, frame):
    print(f"[start_prod] received signal {signum}, shutting down...", flush=True)
    _terminate_children()
    sys.exit(0)


def main():
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    for config in MCP_SERVERS:
        env = dict(os.environ)
        env["HOST"] = config["host"]
        env[config["path_env"].replace("_SERVER_URL", "_PORT")] = str(config["port"])

        cmd = [sys.executable, "-m", config["module"]]
        print(f"[start_prod] starting {config['name']}: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(cmd, env=env)
        _procs.append(proc)

    for config in MCP_SERVERS:
        _wait_ready(config)

    port = os.getenv("PORT", "8501")
    streamlit_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        port,
        "--server.address",
        "0.0.0.0",
    ]
    print(f"[start_prod] launching Streamlit: {' '.join(streamlit_cmd)}", flush=True)
    streamlit_proc = subprocess.Popen(streamlit_cmd, env=os.environ)

    # El supervisor permanece como proceso padre: así los manejadores de
    # señales (SIGTERM/SIGINT) siguen activos y pueden limpiar TODOS los hijos
    # (incluidos los servidores MCP) cuando Render detiene el servicio.
    try:
        streamlit_proc.wait()
    except KeyboardInterrupt:
        pass
    except Exception:
        pass

    _terminate_children()


if __name__ == "__main__":
    main()
