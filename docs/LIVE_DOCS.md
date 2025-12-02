# Live Documentation Server - Quick Reference

This guide shows you how to run a live documentation server that automatically rebuilds when you save changes.

## Quick Start

The easiest way to get started:

```bash
# From the repository root
make docs-serve
```

Or from the `docs` directory:

```bash
# Using the shell script
./serve.sh

# Using the Python script
python serve.py
```

## Prerequisites

Install the documentation dependencies first:

```bash
poetry install --with docs
```

## Available Methods

### Method 1: Makefile Target (Recommended)

```bash
# From repository root
make docs-serve
```

- ✅ Simplest method
- ✅ Automatically opens browser
- ✅ Runs on port 8000

### Method 2: Shell Script

```bash
cd docs
./serve.sh [port]
```

**Features:**

- Default port: 8000
- Watches for changes in all documentation files
- Ignores build artifacts and temporary files
- Also watches Python source code for API docs

**Custom port:**

```bash
./serve.sh 8080
```

### Method 3: Python Script

```bash
cd docs
python serve.py [OPTIONS]
```

**Options:**

- `--port PORT`: Port to serve on (default: 8000)
- `--host HOST`: Host to bind to (default: 0.0.0.0)
- `--open`: Automatically open browser

**Examples:**

```bash
# Default settings
python serve.py

# Custom port with auto-open
python serve.py --port 8080 --open

# Localhost only
python serve.py --host 127.0.0.1
```

### Method 4: Direct Command

```bash
cd docs
poetry run sphinx-autobuild . _build/html --port 8000 --open-browser
```

## How It Works

1. **Initial Build**: The server builds the documentation from scratch
2. **Watch Mode**: Monitors all source files for changes (`.md`, `.rst`, `.py`, etc.)
3. **Auto-Rebuild**: When you save a file, it automatically rebuilds only what changed
4. **Live Reload**: Your browser automatically refreshes to show the updates

## What Files Are Watched?

The server watches:

- ✅ All Markdown files (`.md`)
- ✅ All reStructuredText files (`.rst`)
- ✅ Configuration files (`conf.py`, `config.yml`)
- ✅ Python source code in `nemoguardrails/` (for API docs)
- ✅ Static assets (images, CSS, etc.)

Files ignored:

- ❌ Build output (`_build/`)
- ❌ Temporary files (`.swp`, `*~`)
- ❌ Python cache (`__pycache__/`, `*.pyc`)
- ❌ Git files (`.git/`)

## Accessing the Documentation

Once the server starts, open your browser to:

```
http://127.0.0.1:8000
```

Or if you used a custom port:

```
http://127.0.0.1:<your-port>
```

## Stopping the Server

Press `Ctrl+C` in the terminal to stop the server.

## Troubleshooting

### Port Already in Use

If you see an error about the port being in use:

```bash
# Use a different port
./serve.sh 8080
# or
python serve.py --port 8080
```

### Module Not Found: sphinx-autobuild

Install the documentation dependencies:

```bash
poetry install --with docs
```

### Changes Not Reflecting

1. Check the terminal for build errors
2. Try a full rebuild:

   ```bash
   cd docs
   rm -rf _build
   make docs-serve
   ```

### Browser Not Auto-Refreshing

- Make sure you're viewing the page served by the local server (port 8000)
- Some browser extensions may block the live reload WebSocket
- Try a different browser or incognito mode

## Tips

1. **Keep the terminal visible**: You'll see build progress and any errors
2. **Check for errors**: Red text in the terminal indicates build warnings or errors
3. **Multiple files**: The server batches changes, so save multiple files then wait a moment
4. **Clean builds**: If things look wrong, stop the server and delete `_build/` directory

## Advanced Configuration

The scripts automatically configure:

- Ignore patterns for temporary files
- Debounce delay (1 second) to batch rapid changes
- Watch additional directories (Python source code)
- Rebuild only changed files for speed

To customize, edit:

- `docs/serve.sh` (bash script)
- `docs/serve.py` (Python script)

Or run `sphinx-autobuild` directly with your own options:

```bash
sphinx-autobuild [SOURCE] [BUILD] [OPTIONS]
```

See `sphinx-autobuild --help` for all available options.
