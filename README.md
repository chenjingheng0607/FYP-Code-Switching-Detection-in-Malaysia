# FYP-Code-Switching-Detection-in-Malaysia
FYP project

This repository contains the source code for a Final Year Project (FYP) on **Code-Switching Detection in Malaysian multilingual text**. The project aims to build a system that can identify instances of code-switching between languages commonly used in Malaysia, such as English, Malay, and others.

## Table of Contents

- [About The Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## About The Project

Code-switching is a common linguistic phenomenon in multilingual societies like Malaysia. This project focuses on developing a model to automatically detect code-switched text at the word or sentence level. This can be a foundational step for various downstream NLP tasks such as sentiment analysis, machine translation, and information retrieval in a multilingual context.

This project uses Django, suggesting a web-based interface for demonstrating the detection model.

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

This project requires Python 3.13+ and `uv` for package management.

*   Python (>=3.13)
*   uv

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

This project uses the Django web framework to provide an interface for the code-switching detection model.

1.  **Run the development server:**
    Navigate into the `WebApplication` directory and run the server.
    ```sh
    cd src
    python manage.py runserver
    ```

2.  Open your web browser and go to `http://127.0.0.1:8000` to see your application running.