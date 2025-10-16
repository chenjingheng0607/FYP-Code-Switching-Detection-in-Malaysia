# FYP-Code-Switching-Detection-in-Malaysia
FYP project

This repository contains the source code for a Final Year Project (FYP) on **Code-Switching Detection in Malaysian multilingual text**. The project aims to build a system that can identify instances of code-switching between languages commonly used in Malaysia, such as English, Malay.

## Table of Contents

- [About The Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)

## About The Project

Code-switching is a common linguistic phenomenon in multilingual societies like Malaysia. This project focuses on developing a model to automatically detect code-switched text at the word or sentence level. This can be a foundational step for various downstream NLP tasks such as sentiment analysis, machine translation, and information retrieval in a multilingual context.

This project uses Streamlit to provide a lightweight web interface for demonstrating the detection model.

## Getting Started

To get a local copy up and running, follow these simple steps.

### Installation
1.  Install uv:
    Windows
    ```sh
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

    macOS and Linux
    ```sh
    wget -qO- https://astral.sh/uv/install.sh | sh
    ```

2.  Clone the repo:
    ```sh
    git clone https://github.com/chenjingheng0607/FYP-Code-Switching-Detection-in-Malaysia.git
    cd FYP-Code-Switching-Detection-in-Malaysia
    ```
3.  Create a virtual environment and install dependencies:
    ```sh
    uv sync
    ```

## Usage

This project uses Streamlit to provide an interface for the code-switching detection model.

1.  **Run the Streamlit app:**
    ```sh
    uv run streamlit run app.py
    ```

    If your main Streamlit file has a different name or path, replace `app.py` accordingly (for example: `uv run streamlit run src/ui/main.py`).

2.  Open your web browser and go to `http://localhost:8501` to see the application running.
