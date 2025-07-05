# TIPO-Agent: A Conversational Image Generation System 🤖🎨

![TIPO-Agent Logo](https://img.shields.io/badge/TIPO--Agent-v1.0-blue?style=flat&logo=github)

Welcome to the TIPO-Agent repository! This project focuses on a unique T2I (Text-to-Image) agent system designed for conversational image generation. This system combines a language model (LLM) with T2I capabilities to create images based on conversational inputs. 

**Important Note**: This project is a work in progress (WIP). It is not an AR/In-context MLLM image generation system.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Releases](#releases)

---

## Introduction

TIPO-Agent aims to bridge the gap between language and visual content. By leveraging the power of LLMs and T2I models, users can generate images through natural conversation. This opens new avenues for creativity and interaction, making it easier to visualize ideas and concepts.

## Features

- **Conversational Interface**: Interact with the system in a natural language format.
- **Image Generation**: Generate images based on text prompts provided during the conversation.
- **Customizable Models**: Use different LLM models to suit your needs.
- **Easy Setup**: Simple installation and configuration process.

## Installation

To get started with TIPO-Agent, follow these steps:

1. **Clone the Repository**:
   Use the following command to clone the repository to your local machine:

   ```bash
   git clone https://github.com/joaosz07/TIPO-Agent.git
   cd TIPO-Agent
   ```

2. **Install Requirements**:
   Install the necessary packages. Make sure to install the latest version of `llama-cpp-python`. You may need to build it from source.

   ```bash
   pip install -r requirements.txt
   ```

   Also, install the latest version of KGen, which may require building from source.

3. **Download LLM Model**:
   Download your desired LLM model in the gguf format and place it in the `models/` folder. We recommend using the `mistral-small-3.1` model, as other models have not been extensively tested.

4. **Setup Configuration**:
   Edit the `config.py` file to specify your preferred model and other settings.

## Basic Usage

Once you have completed the installation, you can start using TIPO-Agent. Here’s how:

1. **Clone this repo and install the requirements**:
   Ensure you have completed the installation steps mentioned above.

2. **Download desired LLM model**:
   Place your chosen model in the `models/` folder.

3. **Setup `config.py`**:
   Adjust the configuration file to match your setup.

4. **Run the Application**:
   Start the server with the following command:

   ```bash
   python app.py
   ```

## Configuration

The `config.py` file is crucial for customizing your TIPO-Agent experience. Here’s what you can configure:

- **Model Path**: Specify the path to your LLM model.
- **Image Output Settings**: Set preferences for image resolution and format.
- **Conversational Parameters**: Adjust settings related to how the agent interacts with users.

## Running the Application

To run TIPO-Agent, execute the following command in your terminal:

```bash
python app.py
```

Once the server starts, you can interact with the agent through a conversational interface. Simply input your text prompts, and the system will generate images based on your input.

## Contributing

We welcome contributions to TIPO-Agent! If you would like to contribute, please follow these steps:

1. **Fork the Repository**: Click on the "Fork" button at the top right corner of the page.
2. **Create a New Branch**: Use the following command to create a new branch for your feature or fix:

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**: Implement your changes and commit them with a clear message.

4. **Push Changes**: Push your changes to your forked repository:

   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**: Go to the original repository and create a pull request from your branch.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For questions or feedback, please reach out via the issues section of this repository. We appreciate your interest in TIPO-Agent!

## Releases

To download the latest releases, visit the [Releases](https://github.com/joaosz07/TIPO-Agent/releases) section of this repository. You can find important updates and files there.

---

Feel free to explore and contribute to TIPO-Agent! Your feedback is valuable in shaping this project. Happy coding!