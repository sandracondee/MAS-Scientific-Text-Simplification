# Sistema Multi-agente para la Simplificación de Textos Científicos

Sistema basado en **LLMs y arquitecturas multi-agente** para la simplificación de textos científicos y biomédicos. El objetivo es transformar textos especializados en versiones más claras y accesibles para público no experto, manteniendo la información relevante y la precisión de los contenidos originales.

Para ello, varios agentes especializados colaboran en distintas etapas del proceso: generación de propuestas de simplificación, selección de la mejor versión, evaluación de la legibilidad y fidelidad factual, corrección iterativa y explicación de términos médicos complejos.

La arquitectura está implementada con **LangGraph** y utiliza **MCP (Model Context Protocol)** para integrar herramientas encargadas del cálculo de métricas de simplificación y la búsqueda de definiciones médicas.

![Python Version](https://img.shields.io/badge/python->=3.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-In%20Development-yellow)

## Tabla de contenidos

* [Arquitectura del sistema](#arquitectura-del-sistema)
* [Tecnologías](#tecnologías)
* [Requisitos](#requisitos)
* [Instalación](#instalación)
* [Interfaz de la aplicación](#interfaz-de-la-aplicación)
* [Limitaciones y trabajo futuro](#limitaciones-y-trabajo-futuro)
* [Contribuir](#contribuir)
* [Licencia](#licencia)

## Arquitectura del sistema

El sistema implementa un flujo de trabajo multi-agente orquestado mediante **LangGraph**, en el que cada agente tiene una responsabilidad específica dentro del proceso de simplificación.

![Agentic workflow](assets/agent-workflow.png)

### Flujo de trabajo

1. **Guardrails**: Comprueba que el texto introducido pertenece al dominio médico o biomédico.

2. **Parallel Drafters**: Genera cuatro propuestas de simplificación de forma paralela utilizando diferentes modelos de lenguaje.

3. **Judge**: Analiza las propuestas y selecciona la que considera más adecuada de acuerdo con criterios de Plain Language, naturalidad y cohesión.

4. **Evaluators**: La propuesta seleccionada pasa por dos agentes especializados:

   * **Fact Checker**: Comprueba que la simplificación mantiene la información clínica y numérica del texto original, evitando omisiones y contenido no presente en la fuente.
   * **Readability Evaluator**: Evalúa la claridad, accesibilidad y legibilidad del texto. Cuando existe una simplificación de referencia, puede utilizar las métricas SARI, BLEU y FKGL mediante una herramienta MCP.

5. **Editor**: Si alguno de los evaluadores detecta problemas, genera una nueva versión teniendo en cuenta el feedback recibido. Este proceso puede repetirse hasta un máximo de tres iteraciones para evitar bucles infinitos.

6. **Term Explainer**: Una vez finalizada la simplificación, identifica los términos médicos que puedan seguir siendo complejos y obtiene sus definiciones mediante una herramienta MCP conectada a un diccionario médico.

Este flujo permite separar las diferentes tareas del proceso y facilita el control de cada etapa de la simplificación.

Para consultar la implementación del workflow:

[`src/graph/workflow.py`](src/graph/workflow.py)

La implementación de los diferentes agentes se encuentra en:

[`src/agents/`](src/agents/)

## Tecnologías

Las principales tecnologías utilizadas en el proyecto son:

* **Python**: lenguaje principal del proyecto.
* **LangChain & LangGraph**: integración de modelos de lenguaje y orquestación del sistema multi-agente.
* **Pydantic**: definición y validación de las estructuras de salida de los agentes.
* **MCP / FastMCP**: integración de herramientas externas con los agentes.
* **Streamlit**: desarrollo de la interfaz web interactiva.
* **LLMs**: Gemini, Groq (gpt-oss), Mistral y DeepSeek.
* **Evaluación**: SARI, BLEU y FKGL.
* **uv**: gestión del entorno y las dependencias del proyecto.

## Requisitos

* **Python** >= 3.13
* **uv** (recomendado para instalar y ejecutar el proyecto)
* **pip** >= 23.0 (alternativa para la gestión de dependencias)
* **API Keys** de los proveedores de modelos utilizados, configuradas mediante variables de entorno.

### Dependencias principales

* **LangChain & LangGraph**: gestión y orquestación de los agentes.
* **Streamlit**: interfaz web de la aplicación.
* **MCP / FastMCP**: comunicación entre los agentes y las herramientas.
* **Multiple LLM Providers**: integración con diferentes proveedores y familias de modelos.
* **Hugging Face Evaluate / textstat**: cálculo de métricas utilizadas durante la evaluación.

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/sandracondee/Text-Simplification-ISC.git
cd Text-Simplification-ISC
```

### 2. Configurar variables de entorno

Copiar el fichero `.env.example` como `.env`:

```bash
cp .env.example .env
```

A continuación, completar en `.env` las API Keys necesarias para los modelos utilizados por el sistema.

> **Importante:** no compartas ni subas al repositorio tu fichero `.env`.

### 3. Instalar dependencias con `uv` (recomendado)

`uv` es la opción recomendada para gestionar el entorno virtual y las dependencias del proyecto.

```bash
uv sync
```

Una vez instaladas las dependencias, es necesario iniciar los dos servidores MCP en terminales independientes.

**Servidor de métricas:**

```bash
uv run python -m src.mcp.metrics_server
```

**Servidor de búsqueda de términos médicos:**

```bash
uv run python -m src.mcp.search_server
```

Los servidores utilizan los puertos `8020` y `8021`, respectivamente.

Finalmente, en otra terminal, ejecutar la aplicación:

```bash
uv run streamlit run app.py
```

La aplicación estará disponible en:

```text
http://localhost:8501
```

### 4. Instalación alternativa con `pip`

Si no se dispone de `uv`, también es posible configurar el proyecto utilizando las herramientas estándar de Python.

Crear un entorno virtual:

```bash
python -m venv venv
```

Activarlo:

**Linux/macOS**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

Instalar el proyecto y sus dependencias:

```bash
pip install -e .
```

A continuación, iniciar los servidores MCP en terminales independientes:

**Servidor de métricas:**

```bash
python -m src.mcp.metrics_server
```

**Servidor de búsqueda:**

```bash
python -m src.mcp.search_server
```

Y ejecutar la aplicación:

```bash
streamlit run app.py
```

La aplicación estará disponible en:

```text
http://localhost:8501
```

## Despliegue en Render

El sistema se compone de tres procesos: la interfaz **Streamlit** y los dos **servidores MCP** (métricas y búsqueda). Para desplegarlo en el **plan gratuito** de Render (que permite un único *web service*), los tres procesos se ejecutan **dentro de un mismo contenedor Docker**:

```
Render (1 web service, plan free)
└── Contenedor Docker (python:3.13-slim)
    ├── start_prod.py  (supervisor, CMD del Dockerfile)
    │   ├── arranca src.mcp.metrics_server  → 127.0.0.1:8020 (background)
    │   ├── arranca src.mcp.search_server   → 127.0.0.1:8021 (background)
    │   ├── espera a que ambos respondan
    │   └── lanza streamlit run app.py      → $PORT (primer plano)
```

* Los servidores MCP escuchan **solo en `127.0.0.1` dentro del contenedor**, por lo que **no quedan expuestos a Internet**. Solo la interfaz Streamlit es accesible desde fuera.
* La configuración la definen el [Dockerfile](Dockerfile) y el [render.yaml](render.yaml) (Blueprint). No hay que configurar URLs públicas adicionales.

### Pasos para desplegar

1. Crea una cuenta en [Render](https://render.com).
2. Ve a **New → Blueprint** y conecta este repositorio.
3. Render leerá `render.yaml` y creará 1 *web service* (`text-simplification-app`).
4. Configura las **API keys** en el dashboard de Render, en *Environment*:
   `OPENROUTER_API_KEY`, `GROQ_API_KEY`, `MISTRAL_API_KEY`.
   (Gemini y DeepSeek se sirven a través de OpenRouter, por lo que ambos usan `OPENROUTER_API_KEY`.)
   (También puedes personalizar ahí cualquier variable de modelo del `.env.example`.)
5. El despliegue tarda varios minutos en el primer *build* (las dependencias incluyen `evaluate`/`textstat`/`sacrebleu`).

### Requisitos previos

* **Python 3.13**: el proyecto exige `>=3.13` (ver `pyproject.toml`), por lo que el despliegue usa una imagen Docker `python:3.13-slim` en lugar del runtime python nativo de Render (que no llega a 3.13).

### Notas sobre recursos (plan free)

* El plan free de Render da **512 MB RAM / 0.5 CPU**. Todo el sistema comparte esa memoria.
* Para reducir el pico de memoria en el arranque, el servidor de métricas inicializa `MetricsEvaluator` (que carga `evaluate` y descarga modelos) **de forma perezosa**, es decir, solo la primera vez que se solicita una métrica con texto de referencia.
* Si el plan free se queda corto de RAM con las métricas activas, se puede subir de plan manteniendo el mismo único servicio.

## Interfaz de la aplicación

La aplicación proporciona una interfaz web desarrollada con **Streamlit** desde la que se puede introducir un texto médico o seleccionar uno de los ejemplos disponibles.

![Interfaz principal](assets/interfaz-principal.png)

Una vez finalizado el proceso, se muestra la versión simplificada junto con los términos médicos complejos identificados y sus correspondientes definiciones.

![Salida del sistema](assets/salida-sistema.png)

También es posible consultar el proceso seguido por los agentes, incluyendo las evaluaciones y el feedback generado durante las diferentes etapas del workflow.

![Razonamiento de los agentes](assets/razonamiento-agentes.png)

## Limitaciones y trabajo futuro

Actualmente, el sistema presenta algunas limitaciones:

* La simplificación se realiza únicamente sobre **texto**, sin procesar tablas, gráficos o figuras.
* El sistema está diseñado principalmente para trabajar con **abstracts médicos y biomédicos**.
* La utilización de APIs externas introduce restricciones relacionadas con límites de peticiones y tokens.
* La evaluación de la simplificación no puede determinar completamente aspectos subjetivos como la naturalidad o la adecuación del texto a las necesidades concretas de cada usuario.

Entre las principales líneas de trabajo futuro se encuentran:

* Ampliar el sistema a otros dominios científicos y profesionales.
* Procesar artículos científicos completos.
* Incorporar información procedente de imágenes, tablas y gráficos.
* Integrar herramientas de búsqueda web para complementar las definiciones médicas.
* Evaluar el sistema con usuarios reales.
* Personalizar el nivel de simplificación según el perfil del lector.

## Contribuir

Las contribuciones son bienvenidas.

Para contribuir al proyecto:

1. Realiza un **Fork** del repositorio.
2. Crea una rama para tu nueva funcionalidad:

```bash
git checkout -b feature/mi-mejora
```

3. Realiza los cambios y crea un commit:

```bash
git commit -m "Agregar: descripción de mejora"
```

4. Sube la rama al repositorio:

```bash
git push origin feature/mi-mejora
```

5. Abre un **Pull Request** describiendo los cambios realizados.

## Licencia

Este proyecto se distribuye bajo la licencia **MIT**. Consulta [`LICENSE`](LICENSE) para obtener más información.

---

Para reportar errores, proponer mejoras o realizar sugerencias, abre un [issue](https://github.com/sandracondee/Text-Simplification-ISC/issues) en el repositorio.
