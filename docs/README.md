# Documentation

## Product Documentation

Product documentation for the toolkit is available at
<https://docs.nvidia.com/nemo/guardrails>.

## Building the Documentation

1. Make sure you installed the `docs` dependencies.
   Refer to [CONTRIBUTING.md](../CONTRIBUTING.md) for more information about Poetry and dependencies.

   ```console
   poetry install --with docs
   ```

1. Build the documentation:

   ```console
   make docs
   ```

   The HTML is created in the `_build/docs` directory.

## Live Documentation Server

For local development with automatic rebuilding on file changes, use one of the following methods:

### Option 1: Using the Shell Script (Recommended for Unix/Mac)

```bash
cd docs
./serve.sh [port]
```

Default port is 8000. The server will automatically rebuild documentation when you save changes to any source file.

### Option 2: Using the Python Script (Cross-Platform)

```bash
cd docs
python serve.py [--port PORT] [--host HOST] [--open]
```

Options:

- `--port PORT`: Port to serve on (default: 8000)
- `--host HOST`: Host to bind to (default: 0.0.0.0)
- `--open`: Automatically open browser

Examples:

```bash
# Start server on default port (8000)
python serve.py

# Start server on custom port with auto-open browser
python serve.py --port 8080 --open

# Start server accessible only from localhost
python serve.py --host 127.0.0.1
```

### Option 3: Direct sphinx-autobuild Command

```bash
cd docs
sphinx-autobuild . _build/html --port 8000 --open-browser
```

Once the server is running:

- Open your browser to `http://127.0.0.1:8000`
- Edit any documentation file (`.md`, `.rst`, `.py` configs)
- Save the file
- The browser will automatically refresh with the updated content

Press `Ctrl+C` to stop the server.

## Publishing the Documentation

Tag the commit to publish with `docs-v<semver>`.
Push the tag to GitHub.

To avoid publishing the documentation as the latest, ensure the commit has `/not-latest` on a single line, tag that commit, and push to GitHub.
