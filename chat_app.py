import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

import streamlit as st

# --- Configuration ---
# Set the default URL for the Ollama API here.
DEFAULT_OLLAMA_BASE_URL = "http://172.22.128.1:11434"

def initialize_session_state() -> None:
    """Ensure session state variables exist."""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "ollama_models" not in st.session_state:
        st.session_state.ollama_models = []
    if "performance_metrics" not in st.session_state:
        st.session_state.performance_metrics = None
    if "show_performance_panel" not in st.session_state:
        st.session_state.show_performance_panel = True


def get_ollama_models(base_url: str, timeout_s: int = 5) -> List[str]:
    """Get a list of available models from the Ollama API."""
    try:
        url = base_url.rstrip("/") + "/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [model["name"] for model in data.get("models", [])]
    except Exception:
        return []


def render_header() -> None:
    st.set_page_config(page_title="FYP Chat", page_icon="💬", layout="wide")
    st.title("Chat 💬")
    st.caption("A Streamlit chat page with Ollama integration and performance metrics.")


def render_settings_sidebar() -> None:
    with st.sidebar:
        st.header("⚙️ Settings")
        st.write("Configure Ollama connection and model.")
        
        st.checkbox("Show Performance Panel", key="show_performance_panel")
        
        st.text_input(
            "Ollama base URL",
            key="ollama_base_url",
            value=st.session_state.get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL)
        )

        if st.button("Refresh Models"):
            base_url = st.session_state.get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL)
            st.session_state.ollama_models = get_ollama_models(base_url)
            if not st.session_state.ollama_models:
                st.error(f"Cannot reach Ollama at {base_url}. Please check the URL and ensure Ollama is running.")
            st.rerun()

        if st.session_state.ollama_models:
            st.selectbox("Select a model", st.session_state.ollama_models, key="ollama_model")
        else:
            st.warning("No models found. Press 'Refresh Models' to fetch them.")

        if st.button("Clear chat"):
            st.session_state.chat_messages = []
            st.session_state.performance_metrics = None
            st.rerun()


def render_performance_sidebar() -> None:
    st.header("🚀 Performance")
    metrics = st.session_state.get("performance_metrics")
    if metrics:
        st.write("Metrics for the last generated response:")

        total_duration_ns = metrics.get('total_duration', 0)
        eval_count = metrics.get('eval_count', 0)

        if total_duration_ns > 1e9:
            total_duration_s = total_duration_ns / 1e9
            st.metric(label="Total Duration", value=f"{total_duration_s:.2f} s")
        else:
            total_duration_ms = total_duration_ns / 1e6
            st.metric(label="Total Duration", value=f"{total_duration_ms:.2f} ms")
        
        if total_duration_ns > 0 and eval_count > 0:
            tokens_per_second = eval_count / (total_duration_ns / 1e9)
            st.metric(label="Tokens per Second", value=f"{tokens_per_second:.2f} t/s")
        else:
            st.metric(label="Tokens per Second", value="N/A")
            
        st.metric(label="Tokens Evaluated", value=str(eval_count))

        with st.expander("Show Raw Metrics"):
            st.json(metrics)
    else:
        st.info("Performance metrics will be displayed here after the first message is sent.")


def call_ollama_chat(base_url: str, model: str, messages: List[Dict[str, str]], timeout_s: int = 120) -> Optional[Dict[str, Any]]:
    """Call Ollama's chat API and return the full response dictionary."""
    url = base_url.rstrip("/") + "/api/chat"
    payload: Dict[str, Any] = { "model": model, "messages": messages, "stream": False }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as http_err:
        detail = http_err.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {http_err.code} from Ollama: {detail}") from http_err
    except urllib.error.URLError as url_err:
        raise RuntimeError(f"Failed to reach Ollama at {url}: {url_err}") from url_err


def main() -> None:
    render_header()
    initialize_session_state()

    if not st.session_state.ollama_models:
        base_url = st.session_state.get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL)
        st.session_state.ollama_models = get_ollama_models(base_url)

    render_settings_sidebar()

    # --- Layout Definition ---
    if st.session_state.show_performance_panel:
        main_col, performance_col = st.columns([2, 1])
        with performance_col:
            render_performance_sidebar()
    else:
        main_col = st.container()

    # --- Main Chat Interface ---
    with main_col:
        # Display existing chat messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
                st.markdown(message["content"])

        # Chat input is rendered last and docks to the bottom
        if user_prompt := st.chat_input("Type a message…"):
            # Append and display the user's message
            st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user", avatar="🧑"):
                st.markdown(user_prompt)

            # Generate and display the assistant's response
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking…"):
                    try:
                        base_url = st.session_state.get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL)
                        model = st.session_state.get("ollama_model")
                        
                        if not model:
                            st.error("Please select a model from the settings sidebar.")
                            st.stop()
                        
                        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_messages]
                        full_response = call_ollama_chat(base_url, model, history)
                        
                        if full_response:
                            bot_reply = full_response.get("message", {}).get("content", "")
                            st.session_state.performance_metrics = {k: v for k, v in full_response.items() if k != 'message'}
                            st.session_state.chat_messages.append({"role": "assistant", "content": bot_reply})
                            st.rerun() # Rerun to update performance panel and keep a clean script flow
                        else:
                            st.error("Received an empty response from Ollama.")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()