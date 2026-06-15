/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

declare const React: unknown;

const BUTTON_LABEL = "Copy Starter Prompt";

const STARTER_PROMPT = [
  "# NVIDIA NeMo Guardrails Library Agent Instructions",
  "",
  "You are helping me use the NVIDIA NeMo Guardrails library from this AI coding agent.",
  "Assume I may have installed the Python package with pip and may not have cloned the GitHub repository, so local `.agents/skills/` and `AGENTS.md` files might not exist.",
  "",
  "## How to Help Me",
  "",
  "- Help me add, configure, evaluate, debug, or deploy guardrails for an LLM application.",
  "- Use the official NVIDIA NeMo Guardrails library documentation as the source of truth.",
  "- Prefer the docs MCP server if this agent supports MCP.",
  "- Otherwise, use the documentation index at `https://docs.nvidia.com/nemo/guardrails/llms.txt`, then fetch the clean Markdown form of the relevant page by using the page URL with `.md`.",
  "- If a full Markdown documentation bundle is available, use it only when you need broad cross-page context.",
  "- Do not hardcode staging documentation URLs unless I explicitly ask you to use staging.",
  "- If I have the package installed, check my installed `nemoguardrails` version and use matching versioned docs when available. If you cannot determine the version, ask whether to use the latest docs.",
  "- If I am working from a cloned repository, you may also use local `docs/**/*.mdx`, `README.md`, `CONTRIBUTING.md`, and `AGENTS.md` files as context.",
  "",
  "## Start by Understanding My Goal",
  "",
  "Ask one focused question first: what am I trying to do?",
  "Offer these choices when useful:",
  "",
  "1. Install the library or verify my environment.",
  "2. Add basic input/output guardrails to an app.",
  "3. Choose which guardrail type or catalog item to use.",
  "4. Write or debug Colang flows.",
  "5. Integrate with Python, LangChain, LangGraph, or the Guardrails API server.",
  "6. Add custom actions or a custom model/provider.",
  "7. Evaluate guardrails or run vulnerability scanning.",
  "8. Configure tracing, metrics, logging, Docker, or deployment.",
  "9. Troubleshoot an error.",
  "",
  "## Security and Credentials",
  "",
  "- Never ask me to paste real API keys, tokens, passwords, or private credentials into chat.",
  "- Use placeholders such as `<NVIDIA_API_KEY>`, `<OPENAI_API_KEY>`, or `<YOUR_ENDPOINT>` in examples.",
  "- If a command needs a secret, explain where the secret should be set locally, then let me provide it through my shell, environment, secret manager, or local UI.",
  "- Do not print real secrets in commands, summaries, logs, or generated files.",
  "",
  "## Working Style",
  "",
  "- Keep answers task-oriented and concise.",
  "- Show the smallest working example first, then explain optional production hardening.",
  "- When writing code or configuration, prefer current documented patterns.",
  "- When using live model endpoints in examples, clearly state that unit tests should mock LLM/provider calls.",
  "- If I am contributing to the repository rather than just using the library, switch to the repository contribution rules from `CONTRIBUTING.md` and `AGENTS.md`.",
  "",
  "Begin by asking what I want to do with the NVIDIA NeMo Guardrails library.",
].join("\n");

let resetCopyButtonTimer: ReturnType<typeof setTimeout> | null = null;

export function StarterPromptButton() {
  return (
    <button
      aria-label="Copy NVIDIA NeMo Guardrails library starter prompt"
      aria-live="polite"
      onClick={handleCopyClick}
      style={{
        alignItems: "center",
        background: "#76B900",
        border: "0",
        borderRadius: "8px",
        color: "#111827",
        cursor: "pointer",
        display: "inline-flex",
        fontSize: "0.95rem",
        fontWeight: 700,
        gap: "0.5rem",
        margin: "0.5rem 0 1.5rem",
        padding: "0.75rem 1rem",
        transition: "background 180ms ease, box-shadow 180ms ease, transform 180ms ease",
        willChange: "transform",
      }}
      type="button"
    >
      <svg
        aria-hidden="true"
        focusable="false"
        height="18"
        style={{ flexShrink: 0 }}
        viewBox="0 0 24 24"
        width="18"
      >
        <g data-starter-prompt-icon="prompt">
          <rect
            fill="none"
            height="16"
            rx="3"
            stroke="currentColor"
            strokeWidth="2"
            width="20"
            x="2"
            y="4"
          />
          <path d="M7 9l3 3-3 3" fill="none" stroke="currentColor" strokeWidth="2" />
          <path d="M12 15h5" fill="none" stroke="currentColor" strokeWidth="2" />
        </g>
        <g data-starter-prompt-icon="check" style={{ display: "none" }}>
          <circle cx="12" cy="12" fill="none" r="9" stroke="currentColor" strokeWidth="2" />
          <path d="M8 12.5l2.5 2.5L16 9" fill="none" stroke="currentColor" strokeWidth="2" />
        </g>
      </svg>
      <span data-starter-prompt-label>{BUTTON_LABEL}</span>
    </button>
  );
}

async function handleCopyClick(event: { currentTarget: HTMLButtonElement }) {
  const button = event.currentTarget;
  lockButtonWidth(button);
  setCopyButtonState(button, "Copying...", "#8DD600", "Copying prompt");

  const copied = await copyText(STARTER_PROMPT);
  setCopyButtonState(
    button,
    copied ? "Copied to Clipboard" : "Copy Failed. Try Again",
    copied ? "#8DD600" : "#F97316",
    copied ? "Copied NVIDIA NeMo Guardrails library starter prompt" : "Could not copy starter prompt",
    copied ? "check" : "prompt",
  );
}

async function copyText(text: string): Promise<boolean> {
  if (typeof navigator !== "undefined" && navigator.clipboard) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // Fall through to the textarea fallback for browsers that block clipboard writes.
    }
  }

  if (typeof document === "undefined") {
    return false;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "true");
  textarea.style.position = "fixed";
  textarea.style.top = "-1000px";
  document.body.appendChild(textarea);
  textarea.select();
  try {
    return document.execCommand("copy");
  } finally {
    document.body.removeChild(textarea);
  }
}

function setCopyButtonState(
  button: HTMLButtonElement,
  label: string,
  background: string,
  ariaLabel: string,
  icon: "prompt" | "check" = "prompt",
) {
  if (resetCopyButtonTimer) {
    clearTimeout(resetCopyButtonTimer);
  }

  setButtonLabel(button, label);
  setButtonIcon(button, icon);
  button.setAttribute("aria-label", ariaLabel);
  button.style.background = background;
  button.style.boxShadow = "0 0 0 4px rgb(118 185 0 / 20%)";

  if (typeof button.animate === "function") {
    button.animate(
      [
        { transform: "scale(1)", offset: 0 },
        { transform: "scale(1.04)", offset: 0.45 },
        { transform: "scale(1)", offset: 1 },
      ],
      { duration: 360, easing: "ease-out" },
    );
  }

  resetCopyButtonTimer = setTimeout(() => {
    setButtonLabel(button, BUTTON_LABEL);
    setButtonIcon(button, "prompt");
    button.setAttribute("aria-label", "Copy NVIDIA NeMo Guardrails library starter prompt");
    button.style.background = "#76B900";
    button.style.boxShadow = "none";
    button.style.width = "";
  }, 2000);
}

function setButtonIcon(button: HTMLButtonElement, icon: "prompt" | "check") {
  const promptIcon = button.querySelector<SVGGElement>("[data-starter-prompt-icon='prompt']");
  const checkIcon = button.querySelector<SVGGElement>("[data-starter-prompt-icon='check']");
  if (promptIcon) {
    promptIcon.style.display = icon === "prompt" ? "" : "none";
  }
  if (checkIcon) {
    checkIcon.style.display = icon === "check" ? "" : "none";
  }
}

function setButtonLabel(button: HTMLButtonElement, label: string) {
  const labelElement = button.querySelector<HTMLElement>("[data-starter-prompt-label]");
  if (labelElement) {
    labelElement.textContent = label;
  }
}

function lockButtonWidth(button: HTMLButtonElement) {
  if (!button.style.width) {
    button.style.width = `${button.offsetWidth}px`;
  }
}
