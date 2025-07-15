# CONTRIBUTING GUIDELINES

Welcome to the NeMo Guardrails contributing guide. We're excited to have you here and grateful for your contributions. This document provides guidelines and instructions for contributing to this project.

> [!WARNING]
> We have recently migrated to using Poetry for dependency management and packaging. Please ensure you have Poetry installed and use it for all dependency management tasks.

## Table of Contents

- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements and New Features](#suggesting-enhancements-and-new-features)
  - [Code Contributions](#code-contributions)
    - [Getting Started](#getting-started)
    - [Contribution Workflow](#contribution-workflow)
    - [Pull Request Checklist](#pull-request-checklist)
    - [Folder Structure](#folder-structure)
    - [Coding Style](#coding-style)
    - [Submitting Your Work](#submitting-your-work)
- [Community and Support](#community-and-support)

# How to Contribute

You can contribute to this project in several ways, including:

- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements and New Features](#suggesting-enhancements-and-new-features)
- [Documentation Improvements](#documentation-improvements)
- [Code Contributions](#code-contributions)

## Reporting Bugs

### Steps to Review Before Reporting a Bug

When preparing to report a bug, please follow these steps to ensure efficiency:

- **Review Existing Issues**: Search the [issue tracker](https://github.com/NVIDIA/NeMo-Guardrails/issues) to confirm that the problem you’re experiencing has not been reported already.
- **Confirm the Nature of the Issue**: Ensure that what you are reporting is a genuine bug, not a support question or topic better suited for our [Discussions](https://github.com/NVIDIA/NeMo-Guardrails/discussions) page.
- **Reopen Related Issues**: If you discover a closed issue that mirrors your current experience, create a new issue and reference the closed one with a link to provide context.
- **Check Release Updates**: Look at the latest release notes to see if your issue is mentioned, along with any upgrade instructions or known issues.

### Documenting the Problem Clearly and Thoroughly

To ensure your issue report is easy to find and understand, follow these steps:

- **Create a Clear, Descriptive Title**: Choose a concise and specific title that identifies the problem.
- **Detailed Reproduction Steps**: Provide a step-by-step guide to reproduce the issue. Include all necessary details to avoid ambiguity. Can you reproduce the issue following these steps?
- **Observed vs. Expected Behavior**: Describe what actually happened when you followed the reproduction steps, and explain why this behavior is problematic. Additionally, outline what you expected to happen and why this would be the correct behavior.
- **Minimal Configuration**: Share a minimal configuration that triggers the issue. If your configuration contains information that you don't like to remain on the repo, consider providing it in a [Gist](https://gist.github.com/) or an example repository after redacting any private data (e.g., private package repositories or specific names).
- **Reproducibility Details**: If the issue is intermittent, specify how often it occurs and under what conditions it typically happens.

**Additional Context to Include**:

- **Recent Onset vs. Longstanding Issue**: Clarify whether the issue started recently (e.g., after an update) or has been persistent. If it started recently, check if you can reproduce the issue in an older version, and specify the most recent version where it did not occur.
- **Configuration and Environment Details**:

- The version of NeMo Guardrails you are using (e.g., `nemoguardrails --version`).
- The Python version in use.
- The name and version of the operating system (e.g., Ubuntu 22.04, macOS 14.2).

> **Note**: These information are requested in the template while you are reporting the issue.

**Ensuring Accurate Reproduction Steps**:

To maximize the chances of others understanding and reproducing your issue:

- Test the issue in a clean environment.

This thorough approach helps rule out local setup issues and assists others in accurately replicating your environment for further analysis.

## Suggesting Enhancements and New Features

This section provides instructions on how to submit enhancement or feature suggestions for NeMo Guardrails, whether they involve brand-new features or improvements to current functionality. By following these guidelines, you help maintainers and the community better understand your suggestion and identify any related discussions.

Before Submitting a Suggested Enhancement

- **Review Existing Issues**: Ensure that your suggestion has not already been submitted by checking the [issue tracker](https://github.com/NVIDIA/NeMo-Guardrails/issues) for similar ideas or proposals.

### How to Submit an Enhancement Suggestion?

Enhancement suggestions for NeMo Guardrails should be submitted through the main [issue tracker](https://github.com/NVIDIA/NeMo-Guardrails/issues), using the corresponding issue template provided. Follow these guidelines when submitting:

- **Create a Clear, Descriptive Title**: Choose a title that clearly identifies the nature of your enhancement.
- **Detailed Description**: Provide a comprehensive description of the proposed enhancement. Include specific steps, examples, or scenarios that illustrate how the feature would work or be implemented.
- **Current vs. Proposed Behavior**: Describe the existing behavior or functionality and explain how you would like it to change or be improved. Clarify why this new behavior or feature is beneficial to users and the project.

By providing clear and detailed information, you make it easier for maintainers and the community to assess and discuss your proposal.

## Documentation Improvements

Improving the project documentation is a valuable way to contribute to NeMo Guardrails. By enhancing the documentation, you help users understand the project better, learn how to use it effectively, and contribute to the project more easily. You can contribute to the documentation in several ways:

- **Fixing Typos and Grammar**: If you notice any typos, grammatical errors, or formatting issues in the documentation, feel free to correct them.
- **Clarifying Content**: If you find sections of the documentation that are unclear or confusing, you can propose changes to make them more understandable.
- **Adding Examples**: Providing examples and use cases can help users better understand how to use the project effectively.
- **New Content**: Creating new content such as tutorials, FAQs, Troubleshooting, etc.

## Code Contributions

If you’re contributing for the first time and are searching for an issue to work on, we encourage you to check the [Contributing page](https://github.com/NVIDIA/NeMo-Guardrails/contribute) for suitable candidates. We strive to keep a selection of issues curated for first-time contributors, but sometimes there may be delays in updating. If you don’t find anything that fits, don’t hesitate to ask for guidance.
If you would like to take on an issue, feel free to comment on the issue. We are more than happy to discuss solutions on the issue.

> **Note**: Before submitting a pull request, ensure that you have read and understood the [Contribution Workflow](#contribution-workflow) section. Always open an issue before submitting a pull request so that others can access it in future and potentially discuss the changes you plan to make. We do not accept pull requests without an associated issue.

### Getting Started

To get started quickly, follow the steps below.

1. Ensure you have Python 3.9+ and [Git](https://git-scm.com/) installed on your system. You can check your Python version by running:

   ```bash
   python --version
   # or
   python3 --version
   ```

   > Note: we suggest you use `pyenv` to manage your Python versions. You can find the installation instructions [here](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation).

   Also install `g++` and `python3-dev` packages as dependencies to Annoy.

2. Clone the project repository:

   ```bash
   git clone https://github.com/NVIDIA/NeMo-Guardrails.git
   ```

3. Navigate to the project directory:

   ```bash
   cd nemoguardrails
   ```

4. we use `Poetry` to manage the project dependencies. To install Poetry follow the instructions [here](https://python-poetry.org/docs/#installation):

> Note: This project requires Poetry version >=1.8,<2.0. Please ensure you are using a compatible version before running any Poetry commands.

  Ensure you have `poetry` installed:

   ```bash
   poetry --version
   ```

6. Install the dev dependencies:

   ```bash
   poetry install --with dev
   ```

   The preceding command installs pre-commit, pytest, and other development tools.
   Specify `--with dev,docs` to add the dependencies for building the documentation.

7. If needed, you can install extra dependencies as below:

    ```bash
    poetry install --extras "openai tracing"
    # or Alternatively using the following command
    poetry install -E openai -E tracing

    ```

    to install all the extras:

    ```bash
    poetry install --all-extras
    ```

> **Note**: `dev` is not part of the extras but it is an optional dependency group, so you need to install it as instructed above.

7. Set up pre-commit hooks:

   ```
   pre-commit install
   ```

   This will ensure that the pre-commit checks, including Black, are run before each commit.

8. Run the tests:

   ```bash
   poetry run pytest
   ```

   This will run the test suite to ensure everything is set up correctly.

> **Note**: You should use `poetry run` to run commands within the virtual environment. If you want to avoid prefixing commands with `poetry run`, you can activate the environment using `poetry shell`. This will start a new shell with the virtual environment activated, allowing you to run commands directly.

### Contribution Workflow

This project follows the [GitFlow](https://nvie.com/posts/a-successful-git-branching-model/) branching model which involves the use of several branch types:

- `main`: Latest stable release branch.
- `develop`: Development branch for integrating features.
- `feature/...`: Feature branches for new features and non-emergency bug fixes.
- `release/...`: Release branches for the final versions published to PyPI.
- `hotfix/...`: Hotfix branches for emergency bug fixes.

Additionally, we recommend the use of `docs/...` documentation branches for contributions that update only the project documentation. You can find a comprehensive guide on using GitFlow here: [GitFlow Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow).

To contribute your work, follow the following process:

1. **Fork the Repository**: Fork the project repository to your GitHub account.
2. **Clone Your Fork**: Clone your fork to your local machine.
3. **Create a Feature Branch**: Create a branch from the `develop` branch.
4. **Develop**: Make your changes locally and commit them.
5. **Push Changes**: Push your changes to your GitHub fork.
6. **Open a Pull Request (PR)**: Create a PR against the main project's `develop` branch.

### Pull Request Checklist

Before submitting your Pull Request (PR) on GitHub, please ensure you have completed the following steps. This checklist helps maintain the quality and consistency of the codebase.

1. **Documentation**:

    Ensure that all new code is properly documented. Update the README, API documentation, and any other relevant documentation if your changes introduce new features or change existing functionality.

2. **Tests Passing**:

    Run the project's test suite to make sure all tests pass. Include new tests if you are adding new features or fixing bugs. If applicable, ensure your code is compatible with different Python versions or environments.

    You can run the tests using `pytest`:

    ```bash
    poetry run pytest
    ```

    Or using `make`:

    ```bash
    make tests
    ```

    You can use `tox` to run the tests for the supported Python versions:

    ```bash
    tox
    ```

    We recommend you to run the test coverage to ensure that your changes are well tested:

    ```bash
    make test_coverage
    ```

3. **Changelog Updated**:

    Update the `CHANGELOG.md` file with a brief description of your changes, following the existing format. This is important for keeping track of new features, improvements, and bug fixes.

    > **Note**: If your new feature concerns Colang, please update the `CHANGELOG_Colang.md` file.

4. **Code Style and Quality**:

    Adhere to the project's coding style guidelines. Keep your code clean and readable.

5. **Commit Guidelines**:

    Follow the commit message guidelines, ensuring clear and descriptive commit messages. Sign your commits as per the Developer Certificate of Origin (DCO) or GPG-sign them for verification.

6. **No Merge Conflicts**:

    Before submitting, rebase your branch onto the latest version of the `develop` branch to ensure your PR can be merged smoothly.

7. **Self Review**:

    Self-review your changes and compare them to the contribution guidelines to ensure you haven't missed anything.

By following this checklist, you help streamline the review process and increase the chances of your contribution being merged without significant revisions. Your MR/PR will be reviewed by at least one of the maintainers, who may request changes or further details.

### Folder Structure

The project is structured as follows:

```
.
├── chat-ui
├── docs
├── examples
├── nemoguardrails
├── qa
├── tests
```

- `chat-ui`: includes a static build of the Guardrails Chat UI. This UI is forked from [https://github.com/mckaywrigley/chatbot-ui](https://github.com/mckaywrigley/chatbot-ui) and is served by the NeMo Guardrails server. The source code for the Chat UI is not included as part of this repository.
- `docs`: includes the official documentation of the project.
- `examples`: various examples, including guardrails configurations (example bots, using different LLMs and others), notebooks, or Python scripts.
- `nemoguardrails`: the source code for the main `nemoguardrails` package.
- `qa`: a set of scripts the QA team uses.
- `tests`: the automated tests set that runs automatically as part of the CI pipeline.

### Coding Style

We follow the [Black](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html) coding style for this project. To maintain consistent code quality and style, the [pre-commit](https://pre-commit.com) framework is used. This tool automates the process of running various checks, such as linters and formatters, before each commit. It helps catch issues early and ensures all contributions adhere to our coding standards.

### Setting Up Pre-Commit

1. **Install Pre-Commit**:

    First, you need to install pre-commit on your local machine. It can be installed via `poetry`:

    ```bash
    poetry add pre-commit
    ```

    Alternatively, you can use other installation methods as listed in the [pre-commit installation guide](https://pre-commit.com/#install).

2. **Configure Pre-Commit in Your Local Repository**:

    In the root of the project repository, there should be a [`.pre-commit-config.yaml`](./.pre-commit-config.yaml) file which contains the configuration and the hooks we use. Run the following command in the root of the repository to set up the git hook scripts:

    ```bash
    pre-commit install
    ```

3. **Running Pre-Commit**

   **Automatic Checks**: Once `pre-commit` is installed, the configured hooks will automatically run on each Git commit. If any changes are necessary, the commit will fail, and you'll need to make the suggested changes.

   **Manual Run**: You can manually run all hooks against all the files with the following command:

   ```bash
   pre-commit run --all-files
   ```

    To do steps 2 and 3 in one command:

    ```bash
    make pre_commit
    ```

### Installing Dependencies Without Modifying `pyproject.toml`

To install a dependency using Poetry without adding it to the `pyproject.toml` file, you can use `pip` within the Poetry-managed virtual environment. Here's how to do it:

1. **Activate the Poetry virtual environment**:
   - Run `poetry shell` to activate the virtual environment managed by Poetry.

> **Note**: If you don't want to activate the virtual environment, you can use `poetry run` to run commands within the virtual environment.

2. **Install the package using `pip`**:
   - Once inside the virtual environment, you can use `pip` to install the package without affecting the `pyproject.toml`. For example:

     ```bash
     pip install <package-name>
     # or if the virtual environment is not activated
     poetry run pip install <package-name>
     ```

This will install the package only in the virtual environment without tracking it in `pyproject.toml`.

This method is useful when you need a package temporarily or for personal development tools that you don't want to be part of your project's formal dependencies.

**Important Considerations**:

- Using `pip` directly inside a Poetry-managed environment bypasses Poetry's dependency resolution, so be cautious of potential conflicts with other dependencies.
- This approach does not update the lock file (`poetry.lock`), meaning these changes are not reproducible for others or on different environments unless manually replicated.
- If you decided to add the dependency permanently, you should add it to the `pyproject.toml` file using Poetry's `add` command.

This workaround is commonly used because Poetry currently does not have a built-in feature to install packages without modifying `pyproject.toml`.

## Jupyter Notebook Documentation

For certain features, you can provide documentation in the form of a Jupyter notebook. In addition to the notebook, we also require that you generate a README.md file next to the Jupyter notebook, with the same content. To achieve this, follow the following process:

1. Place the jupyter notebook in a separate sub-folder.

2. Install `nbdoc`:

   ```bash
   poetry run pip install nbdoc
   ```

3. Use the `build_notebook_docs.py` script from the root of the project to perform the conversion:

   ```bash
   poetry run python build_notebook_docs.py PATH/TO/SUBFOLDER
   ```

### Submitting Your Work

We require that all contributions are certified under the terms of the Developer Certificate of Origin (DCO), Version 1.1. This certifies that the contribution is your original work or you have the right to submit it under the same or compatible license. Any public contribution that contains commits that are not signed off will not be accepted.

To simplify the process, we accept GPG-signed commits as fulfilling the requirements of the DCO.

#### Why GPG Signatures?

A GPG-signed commit provides cryptographic assurance that the commit was made by the holder of the corresponding private key. By configuring your commits to be signed by GPG, you not only enhance the security of the repository but also implicitly certify that you have the rights to submit the work under the project's license and agree to the DCO terms.

#### Setting Up Git for Signed Commits

1. **Generate a GPG key pair**:

    If you don't already have a GPG key, you can generate a new GPG key pair by following the instructions here: [Generating a new GPG key](https://docs.github.com/en/authentication/managing-commit-signature-verification/generating-a-new-gpg-key).

2. **Add your GPG key to your GitHub/GitLab account**:

   After generating your GPG key, add it to your GitHub account by following these steps: [Adding a new GPG key to your GitHub account](https://docs.github.com/en/authentication/managing-commit-signature-verification/adding-a-gpg-key-to-your-github-account).

3. **Configure Git to sign commits:**

   Tell Git to use your GPG key by default for signing your commits:

   ```bash
   git config --global user.signingkey YOUR_GPG_KEY_ID
   ```

4. **Sign commits**:

   Sign individual commits using the `-S` flag

   ```bash
   git commit -S -m "Your commit message"
   ```

   Or, enable commit signing by default (recommended):

   ```bash
   git config --global commit.gpgsign true
   ```

**Troubleshooting and Help**: If you encounter any issues or need help with setting up commit signing, please refer to the [GitHub documentation on signing commits](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits).
Feel free to contact the project maintainers if you need further assistance.

#### Developer Certificate of Origin (DCO)

To ensure the quality and legality of the code base, all contributors are required to certify the origin of their contributions under the terms of the Developer Certificate of Origin (DCO), Version 1.1:

  ```
  Developer Certificate of Origin
  Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
  1 Letterman Drive
  Suite D4700
  San Francisco, CA, 94129

  Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.

  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

  (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

  (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

  (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```

#### Why the DCO is Important

The DCO helps to ensure that contributors have the right to submit their contributions under the project's license, protecting both the contributors and the project. It's a lightweight way to manage contributions legally without requiring a more cumbersome Contributor License Agreement (CLA).

#### Summary

- A GPG-signed commit will be accepted as a declaration that you agree to the terms of the DCO.
- Alternatively, you can manually add a "Signed-off-by" line to your commit messages to comply with the DCO.

By following these guidelines, you help maintain the integrity and legal compliance of the project.

## Community and Support

For general questions or discussion about the project, use the [discussions](https://github.com/NVIDIA/NeMo-Guardrails/discussions) section.

Thank you for contributing to NeMo Guardrails!
