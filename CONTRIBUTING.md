# CONTRIBUTING GUIDELINES

Welcome to the NeMo Guardrails contributing guide. We're excited to have you here and grateful for your contributions. This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Getting Started](#getting-started)
- [Contribution Workflow](#contribution-workflow)
- [Pull Request Checklist](#pull-request-checklist)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)
- [Folder Structure](#folder-structure)
- [Coding Style](#coding-style)
- [Submitting Your Work](#submitting-your-work)
- [Community and Support](#community-and-support)

## Getting Started

To get started quickly, follow the steps below.

1. Ensure you have Python 3.9+ and [Git](https://git-scm.com/) installed on your system. You can check your Python version by running:

   ```bash
   python --version
   # or
   python3 --version
   ```

2. Clone the project repository:

   ```bash
   git clone https://github.com/NVIDIA/NeMo-Guardrails.git
   ```

3. Navigate to the project directory:

   ```bash
   cd NeMo-Guardrails
   ```

4. Create a virtual environment to isolate your project's dependencies:

   ```bash
   python3 -m venv venv
   ```

   Replace the second `venv` above with the desired name for your virtual environment directory.

5. Activate the virtual environment:

   - On Windows:

     ```powershell
     venv\Scripts\activate
     ```

   - On macOS and Linux:

     ```bash
     source venv/bin/activate
     ```

6. Install the main dependencies:

   ```bash
   python -m pip install ".[dev]"
   ```

   This will install pre-commit, pytest, and other development tools, as well as all optional dependencies.

7. Set up pre-commit hooks:

   ```
   pre-commit install
   ```

   This will ensure that the pre-commit checks, including Black, are run before each commit.

## Contribution Workflow

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

## Pull Request Checklist

Before submitting your Pull Request (PR) on GitHub, please ensure you have completed the following steps. This checklist helps maintain the quality and consistency of the codebase.

1. **Documentation**:

    Ensure that all new code is properly documented. Update the README, API documentation, and any other relevant documentation if your changes introduce new features or change existing functionality.

2. **Tests Passing**:

    Run the project's test suite to make sure all tests pass. Include new tests if you are adding new features or fixing bugs. If applicable, ensure your code is compatible with different Python versions or environments.

3. **Changelog Updated**:

    Update the CHANGELOG.md file with a brief description of your changes, following the existing format. This is important for keeping track of new features, improvements, and bug fixes.

4. **Code Style and Quality**:

    Adhere to the project's coding style guidelines. Keep your code clean and readable.

5. **Commit Guidelines**:

    Follow the commit message guidelines, ensuring clear and descriptive commit messages. Sign your commits as per the Developer Certificate of Origin (DCO) or GPG-sign them for verification.

6. **No Merge Conflicts**:

    Before submitting, rebase your branch onto the latest version of the `develop` branch to ensure your PR can be merged smoothly.

7. **Self Review**:

    Self-review your changes and compare them to the contribution guidelines to ensure you haven't missed anything.

By following this checklist, you help streamline the review process and increase the chances of your contribution being merged without significant revisions. Your MR/PR will be reviewed by at least one of the maintainers, who may request changes or further details.

## Reporting Bugs

Bugs are tracked as GitHub issues. Create an issue on the repository and clearly describe the issue with as much detail as possible.

## Feature Requests

Feature requests are welcome. To make a feature request, please open an issue on GitHub and tag it as a feature request. Include any specific requirements and why you think it's a valuable addition.

## Folder Structure

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


## Coding Style

We follow the [Black](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html) coding style for this project. To maintain consistent code quality and style, the [pre-commit](https://pre-commit.com) framework is used. This tool automates the process of running various checks, such as linters and formatters, before each commit. It helps catch issues early and ensures all contributions adhere to our coding standards.

### Setting Up Pre-Commit

1. **Install Pre-Commit**:

    First, you need to install pre-commit on your local machine. It can be installed via pip:
    ```bash
    pip install pre-commit
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

## Jupyter Notebook Documentation

For certain features, you can provide documentation in the form of a Jupyter notebook. In addition to the notebook, we also require that you generate a README.md file next to the Jupyter notebook, with the same content. To achieve this, follow the following process:

1. Place the jupyter notebook in a separate sub-folder.

2. Install `nbdoc`:
   ```bash
   pip install nbdoc
   ```

3. Use the `build_notebook_docs.py` script from the root of the project to perform the conversion:

   ```bash
   python build_notebook_docs.py PATH/TO/SUBFOLDER
   ```

## Submitting Your Work

We require that all contributions are certified under the terms of the Developer Certificate of Origin (DCO), Version 1.1. This certifies that the contribution is your original work or you have the right to submit it under the same or compatible license. Any public contribution that contains commits that are not signed off will not be accepted.

To simplify the process, we accept GPG-signed commits as fulfilling the requirements of the DCO.

### Why GPG Signatures?

A GPG-signed commit provides cryptographic assurance that the commit was made by the holder of the corresponding private key. By configuring your commits to be signed by GPG, you not only enhance the security of the repository but also implicitly certify that you have the rights to submit the work under the project's license and agree to the DCO terms.

### Setting Up Git for Signed Commits

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

### Developer Certificate of Origin (DCO)

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

### Why the DCO is Important

The DCO helps to ensure that contributors have the right to submit their contributions under the project's license, protecting both the contributors and the project. It's a lightweight way to manage contributions legally without requiring a more cumbersome Contributor License Agreement (CLA).


### Traditional Sign-Off

For those who prefer or are unable to use GPG-signed commits, we still accept the traditional "Signed-off-by" line in commit messages. To add this line manually, use the `-s' or `--signoff` flag in your commit command:

```bash
git commit -s -m "Your commit message"
```

### Summary

- A GPG-signed commit will be accepted as a declaration that you agree to the terms of the DCO.
- Alternatively, you can manually add a "Signed-off-by" line to your commit messages to comply with the DCO.

By following these guidelines, you help maintain the integrity and legal compliance of the project.


## Community and Support

For general questions or discussion about the project, use the [discussions](https://github.com/NVIDIA/NeMo-Guardrails/discussions) section.

Thank you for contributing to NeMo Guardrails!
