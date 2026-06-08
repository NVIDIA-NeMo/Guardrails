#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Markdown/MDX link checker for Guardrails docs.
#
# Usage:
#   scripts/check-docs-links.sh
#   scripts/check-docs-links.sh --local-only
#   scripts/check-docs-links.sh path/to/page.md path/to/page.mdx
#
# Environment:
#   CHECK_DOC_LINKS_REMOTE            If 0, skip http(s) probes.
#   CHECK_DOC_LINKS_VERBOSE           If 1, log each URL while curling.
#   CHECK_DOC_LINKS_IGNORE_EXTRA      Comma-separated extra http(s) URLs to skip.
#   CHECK_DOC_LINKS_IGNORE_URL_REGEX  Skip remote probes when the full URL matches this ERE.
#   CHECK_DOCS_FERN_NAV_YML           Override docs/index.yml for tests.
#   NODE                              Node binary for Fern route parsing.
#   CURL                              curl binary for remote probes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

CURL="${CURL:-curl}"
NODE="${NODE:-node}"
CHECK_DOC_LINKS_REMOTE="${CHECK_DOC_LINKS_REMOTE:-1}"
VERBOSE="${CHECK_DOC_LINKS_VERBOSE:-0}"
EXTRA_FILES=()

usage() {
  cat <<'EOF'
Markdown/MDX link checker for Guardrails docs.

Usage: scripts/check-docs-links.sh [options] [extra.md/.mdx ...]

Options:
  --local-only  Do not curl http(s) URLs (same as CHECK_DOC_LINKS_REMOTE=0).
  --verbose     Log each URL while curling.
  -h, --help    Show this help.

Environment: CHECK_DOC_LINKS_REMOTE, CHECK_DOC_LINKS_VERBOSE,
  CHECK_DOC_LINKS_IGNORE_EXTRA, CHECK_DOC_LINKS_IGNORE_URL_REGEX,
  CHECK_DOCS_FERN_NAV_YML, NODE, CURL.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local-only)
      CHECK_DOC_LINKS_REMOTE=0
      shift
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_FILES+=("$@")
      break
      ;;
    -*)
      echo "check-docs-links: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      EXTRA_FILES+=("$1")
      shift
      ;;
  esac
done

log() {
  printf '%s\n' "check-docs-links: $*"
}

collect_default_docs() {
  local f
  for f in \
    "$REPO_ROOT/README.md" \
    "$REPO_ROOT/CONTRIBUTING.md" \
    "$REPO_ROOT/SECURITY.md" \
    "$REPO_ROOT/CHANGELOG.md"; do
    [[ -f "$f" ]] && printf '%s\n' "$f"
  done
  if [[ -d "$REPO_ROOT/docs" ]]; then
    find "$REPO_ROOT/docs" \
      -path "$REPO_ROOT/docs/_static/python-sdk-reference" -prune \
      -o -type f \( -name '*.md' -o -name '*.mdx' \) -print | LC_ALL=C sort
  fi
}

extract_targets() {
  LC_ALL=C perl -CS -ne '
    if ($in_fence) {
      if (/^\s*(`{3,}|~{3,})(.*)$/) {
        my $fence = $1;
        my $rest = $2;
        my $char = substr($fence, 0, 1);
        my $length = length($fence);
        if ($char eq $fch && $length >= $flen && $rest =~ /^\s*$/) {
          ($in_fence, $fch, $flen) = (0, "", 0);
        }
      }
      next;
    }

    my $line = $.;
    my $text = $_;
    my $visible = "";

    while (length $text) {
      if ($in_comment) {
        if ($text =~ s/^(.*?)-->//s) {
          $in_comment = 0;
          next;
        }
        $text = "";
        next;
      }

      if ($text =~ s/^(.*?)<!--//s) {
        $visible .= $1;
        $in_comment = 1;
        next;
      }

      if ($text =~ /-->/) {
        die "malformed HTML comment\n";
      }

      $visible .= $text;
      last;
    }

    if ($visible =~ /^\s*(`{3,}|~{3,})(.*)$/) {
      my $fence = $1;
      my $char = substr($fence, 0, 1);
      my $length = length($fence);
      ($in_fence, $fch, $flen) = (1, $char, $length);
      next;
    }

    my $scan = $visible;
    $scan =~ s/`[^`]*`//g;
    while ($scan =~ /\!?\[[^\]]*\]\(([^)\s]+)(?:\s+["'"'"'][^)"'"'"']*["'"'"'])?\)/g) { print $line . "\t" . $1 . "\n"; }
    while ($scan =~ /<(https?:[^>\s]+)>/g) { print $line . "\t" . $1 . "\n"; }
    while ($scan =~ /\bhref=(["'"'"'])([^"'"'"'\s]+)\1/g) { print $line . "\t" . $2 . "\n"; }
    END {
      die "malformed HTML comment\n" if $in_comment;
    }
  ' -- "$1"
}

FERN_ROUTE_INDEX_LOADED=0
FERN_ROUTE_INDEX=""

load_fern_route_index() {
  [[ "$FERN_ROUTE_INDEX_LOADED" -eq 1 ]] && return 0
  FERN_ROUTE_INDEX_LOADED=1

  local nav_yml="${CHECK_DOCS_FERN_NAV_YML:-$REPO_ROOT/docs/index.yml}"
  [[ -f "$nav_yml" ]] || return 0
  if ! command -v "$NODE" >/dev/null 2>&1; then
    return 0
  fi

  local _fern_route_index_err
  _fern_route_index_err="$(mktemp)"
  if ! FERN_ROUTE_INDEX="$(
    "$NODE" - "$REPO_ROOT" "$nav_yml" <<'NODE' 2>"$_fern_route_index_err"
const fs = require("node:fs");
const path = require("node:path");

const repoRoot = process.argv[2];
const navPath = process.argv[3];
const docsRoot = path.join(repoRoot, "docs");
const rows = [];
const routes = new Set();

function clean(value) {
  let out = value.trim();
  const hash = out.indexOf(" #");
  if (hash >= 0) out = out.slice(0, hash).trim();
  if ((out.startsWith('"') && out.endsWith('"')) || (out.startsWith("'") && out.endsWith("'"))) {
    out = out.slice(1, -1);
  }
  return out;
}

function normalizeRoute(input) {
  let out = input.replace(/\\/g, "/").replace(/^\/+/, "");
  out = out.replace(/\.(?:md|mdx)$/i, "");
  out = out.replace(/\/index$/i, "");
  return out;
}

function emit(source, route) {
  route = normalizeRoute(route);
  if (!route) return;
  rows.push(`${source}\t${route}`);
  routes.add(route);
}

function emitDocsIndexRoutes() {
  const lines = fs.readFileSync(navPath, "utf8").split(/\r?\n/);
  let stack = [];
  let current = null;
  for (const line of lines) {
    const itemMatch = line.match(/^(\s*)-\s+(page|section):/);
    if (itemMatch) {
      const indent = itemMatch[1].length;
      while (stack.length && stack[stack.length - 1].indent >= indent) stack.pop();
      current = {
        indent,
        type: itemMatch[2],
        parent: stack.map((part) => part.slug),
        path: "",
        slug: "",
        emitted: false,
        pushed: false,
      };
      continue;
    }

    const propMatch = line.match(/^(\s*)(path|slug):\s*(.+?)\s*$/);
    if (!propMatch || !current) continue;
    const indent = propMatch[1].length;
    if (indent !== current.indent + 2) continue;

    const key = propMatch[2];
    const value = clean(propMatch[3]);
    if (key === "path") current.path = value;
    if (key === "slug") current.slug = value;
    if (!current.emitted && current.path && current.slug) {
      emit(current.path, [...current.parent, current.slug].join("/"));
      current.emitted = true;
    }
    if (!current.pushed && current.type === "section" && current.slug) {
      stack.push({ indent: current.indent, slug: current.slug });
      current.pushed = true;
    }
  }
}

function walk(directory) {
  if (!fs.existsSync(directory)) return;
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const fullPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      walk(fullPath);
    } else if (entry.isFile() && entry.name.endsWith(".mdx")) {
      emitFrontmatterSlug(fullPath);
    }
  }
}

function emitFrontmatterSlug(fullPath) {
  const rel = path.relative(docsRoot, fullPath).replace(/\\/g, "/");
  const content = fs.readFileSync(fullPath, "utf8");
  const match = content.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!match) return;
  const slugMatch = match[1].match(/^slug:\s*(.+?)\s*$/m);
  if (!slugMatch) return;
  emit(rel, clean(slugMatch[1]));
}

function emitGeneratedNavigationRoutes() {
  const navFiles = [
    path.join(docsRoot, "_static", "python-sdk-reference", "_navigation.yml"),
  ];
  for (const generatedNav of navFiles) {
    if (!fs.existsSync(generatedNav)) continue;
    const lines = fs.readFileSync(generatedNav, "utf8").split(/\r?\n/);
    let currentSlug = "";
    for (const line of lines) {
      const slugMatch = line.match(/^\s*slug:\s*(.+?)\s*$/);
      if (slugMatch) {
        currentSlug = clean(slugMatch[1]);
        routes.add(normalizeRoute(currentSlug));
        continue;
      }
      const pageMatch = line.match(/^\s*pageId:\s*(.+?)\s*$/);
      if (pageMatch && currentSlug) {
        emit(`_static/python-sdk-reference/${clean(pageMatch[1])}`, currentSlug);
      }
    }
  }
}

emitDocsIndexRoutes();
emitGeneratedNavigationRoutes();
walk(path.join(docsRoot, "_static", "python-sdk-reference", "guardrails-python-sdk"));

for (const route of routes) {
  rows.push(`\t${route}`);
}

if (rows.length === 0) {
  throw new Error(`no Fern routes found in ${navPath}`);
}
process.stdout.write(rows.join("\n"));
NODE
  )"; then
    echo "check-docs-links: failed to parse Fern navigation ${nav_yml#"$REPO_ROOT"/}: $(tr '\n' ' ' <"$_fern_route_index_err" | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//')" >&2
    rm -f "$_fern_route_index_err"
    return 1
  fi
  rm -f "$_fern_route_index_err"
}

normalize_fern_route_path() {
  local input="$1" part
  input="${input#/}"
  case "$input" in
    nemo/guardrails/latest/*) input="${input#nemo/guardrails/latest/}" ;;
    nemo/guardrails/*) input="${input#nemo/guardrails/}" ;;
    latest/*) input="${input#latest/}" ;;
  esac
  input="${input%.mdx}"
  input="${input%.md}"
  input="${input%/index}"

  local -a parts=() out=()
  local IFS='/'
  read -r -a parts <<<"$input"
  unset IFS
  for part in "${parts[@]}"; do
    case "$part" in
      "" | .) ;;
      ..)
        if [[ "${#out[@]}" -eq 0 ]]; then
          return 1
        fi
        unset 'out[${#out[@]}-1]'
        ;;
      *) out+=("$part") ;;
    esac
  done

  local joined
  joined="$(
    IFS=/
    printf '%s' "${out[*]}"
  )"
  printf '%s' "$joined"
}

fern_route_exists() {
  local route="$1"
  if ! load_fern_route_index; then
    return 3
  fi
  [[ -n "$FERN_ROUTE_INDEX" ]] || return 1

  route="$(normalize_fern_route_path "$route")" || return 1
  while IFS=$'\t' read -r _source indexed_route || [[ -n "${indexed_route:-}" ]]; do
    [[ "$indexed_route" == "$route" ]] && return 0
  done <<<"$FERN_ROUTE_INDEX"
  return 1
}

fern_relative_ref_exists() {
  local md_path="$1" stripped="$2"
  local abs_md="$md_path" source_rel current route base
  [[ "$abs_md" == /* ]] || abs_md="$REPO_ROOT/$abs_md"
  case "$abs_md" in
    "$REPO_ROOT/docs/"*) source_rel="${abs_md#"$REPO_ROOT/docs/"}" ;;
    *) return 1 ;;
  esac

  if ! load_fern_route_index; then
    return 3
  fi
  [[ -n "$FERN_ROUTE_INDEX" ]] || return 1

  while IFS=$'\t' read -r _source current || [[ -n "${current:-}" ]]; do
    [[ "$_source" == "$source_rel" ]] || continue
    base="${current%/*}"
    [[ "$base" == "$current" ]] && base=""
    route="${base:+$base/}$stripped"
    local _fern_rc
    if fern_route_exists "$route"; then
      _fern_rc=0
    else
      _fern_rc=$?
    fi
    if [[ "$_fern_rc" -eq 0 ]]; then
      return 0
    elif [[ "$_fern_rc" -eq 3 ]]; then
      return 3
    fi
  done <<<"$FERN_ROUTE_INDEX"
  return 1
}

source_ref_exists() {
  local base_dir="$1" stripped="$2" candidate
  local -a candidates=("$stripped")
  if [[ "$stripped" == */ ]]; then
    candidates+=("${stripped}index.mdx" "${stripped}index.md")
  else
    candidates+=("$stripped.mdx" "$stripped.md" "$stripped/index.mdx" "$stripped/index.md")
  fi

  for candidate in "${candidates[@]}"; do
    if (cd "$base_dir" && [[ -e "$candidate" ]]); then
      return 0
    fi
  done
  return 1
}

site_source_ref_exists() {
  local stripped="$1"
  local site_path="${stripped#/}"
  local -a site_paths=("$site_path")
  case "$site_path" in
    nemo/guardrails/latest/*) site_paths+=("${site_path#nemo/guardrails/latest/}") ;;
    nemo/guardrails/*) site_paths+=("${site_path#nemo/guardrails/}") ;;
    latest/*) site_paths+=("${site_path#latest/}") ;;
  esac

  local route_path
  for route_path in "${site_paths[@]}"; do
    if source_ref_exists "$REPO_ROOT/docs" "$route_path"; then
      return 0
    fi
  done
  return 1
}

has_markdown_extension() {
  case "$1" in
    *.md | *.mdx) return 0 ;;
    *) return 1 ;;
  esac
}

check_local_ref() {
  local md_path="$1" line_no="$2" target="$3"
  local stripped

  stripped="${target%%\#*}"
  stripped="${stripped%%\?*}"

  [[ -z "$stripped" ]] && return 0
  [[ "$stripped" == api:* ]] && return 0
  [[ "$stripped" == mailto:* ]] && return 0
  [[ "$stripped" == tel:* ]] && return 0
  [[ "$stripped" == javascript:* ]] && return 0

  if [[ "$stripped" == http://* || "$stripped" == https://* ]]; then
    return 2
  fi
  if [[ "$stripped" == *://* ]]; then
    return 0
  fi

  if [[ "$stripped" == /* ]]; then
    if [[ "$stripped" == /guardrails-python-sdk/* ]]; then
      return 0
    fi
    local _fern_rc
    if fern_route_exists "$stripped"; then
      _fern_rc=0
    else
      _fern_rc=$?
    fi
    if [[ "$_fern_rc" -eq 0 ]] && has_markdown_extension "$stripped"; then
      echo "check-docs-links: route-style link should omit .md/.mdx extension in $md_path:$line_no -> $target" >&2
      return 1
    fi
    if [[ "$_fern_rc" -eq 0 ]]; then
      return 0
    elif [[ "$_fern_rc" -eq 3 ]]; then
      return 1
    fi
    echo "check-docs-links: broken site route in $md_path:$line_no -> $target" >&2
    return 1
  fi

  local _fern_relative_rc
  if fern_relative_ref_exists "$md_path" "$stripped"; then
    _fern_relative_rc=0
  else
    _fern_relative_rc=$?
  fi
  if [[ "$_fern_relative_rc" -eq 0 ]] && has_markdown_extension "$stripped"; then
    echo "check-docs-links: route-style link should omit .md/.mdx extension in $md_path:$line_no -> $target" >&2
    return 1
  fi
  if [[ "$_fern_relative_rc" -eq 0 ]]; then
    return 0
  elif [[ "$_fern_relative_rc" -eq 3 ]]; then
    return 1
  fi
  if source_ref_exists "$(dirname "$md_path")" "$stripped"; then
    return 0
  fi
  echo "check-docs-links: broken local link in $md_path:$line_no -> $target" >&2
  return 1
}

check_remote_url() {
  local url="$1"
  if ! command -v "$CURL" >/dev/null 2>&1; then
    echo "check-docs-links: curl not found; cannot verify $url" >&2
    return 1
  fi
  if ! "$CURL" -fsS -L -o /dev/null \
    --connect-timeout 12 --max-time 35 \
    -A 'Guardrails-doc-link-check/1.0 (+https://github.com/NVIDIA-NeMo/Guardrails)' \
    "$url" 2>/dev/null; then
    echo "check-docs-links: unreachable URL: $url" >&2
    return 1
  fi
  return 0
}

normalize_url_for_ignore_match() {
  local u="$1"
  u="${u%%\#*}"
  u="${u%/}"
  printf '%s' "$u"
}

check_docs_default_ignored_urls() {
  printf '%s\n' \
    'https://github.com/NVIDIA-NeMo/Guardrails/commits/develop' \
    'https://github.com/NVIDIA-NeMo/Guardrails/pulls?q=is%3Apr+is%3Amerged' \
    'https://github.com/NVIDIA-NeMo/Guardrails/pulls?q=is:pr+is:merged'
}

url_should_skip_remote_probe() {
  local url="$1"
  local nu ign _re
  nu="$(normalize_url_for_ignore_match "$url")"

  while IFS= read -r ign || [[ -n "${ign:-}" ]]; do
    [[ -z "${ign:-}" ]] && continue
    [[ "$(normalize_url_for_ignore_match "$ign")" == "$nu" ]] && return 0
  done < <(check_docs_default_ignored_urls)

  if [[ -n "${CHECK_DOC_LINKS_IGNORE_EXTRA:-}" ]]; then
    local -a _extra_parts=()
    local IFS=','
    read -ra _extra_parts <<<"${CHECK_DOC_LINKS_IGNORE_EXTRA}"
    unset IFS
    for ign in "${_extra_parts[@]}"; do
      ign="${ign#"${ign%%[![:space:]]*}"}"
      ign="${ign%"${ign##*[![:space:]]}"}"
      [[ -z "$ign" ]] && continue
      [[ "$(normalize_url_for_ignore_match "$ign")" == "$nu" ]] && return 0
    done
  fi

  if [[ -n "${CHECK_DOC_LINKS_IGNORE_URL_REGEX:-}" ]]; then
    _re="${CHECK_DOC_LINKS_IGNORE_URL_REGEX}"
    [[ "$url" =~ $_re ]] && return 0
  fi

  return 1
}

run_links_check() {
  local -a DOC_FILES
  if [[ ${#EXTRA_FILES[@]} -gt 0 ]]; then
    DOC_FILES=("${EXTRA_FILES[@]}")
  else
    DOC_FILES=()
    while IFS= read -r _docf || [[ -n "${_docf:-}" ]]; do
      [[ -z "${_docf:-}" ]] && continue
      DOC_FILES+=("$_docf")
    done < <(collect_default_docs | LC_ALL=C sort -u)
  fi

  if [[ ${#DOC_FILES[@]} -eq 0 ]]; then
    echo "check-docs-links: no Markdown/MDX files to scan under $REPO_ROOT" >&2
    return 1
  fi

  log "repository root: $REPO_ROOT"
  log "scope: README, CONTRIBUTING, SECURITY, CHANGELOG, docs/**/*.{md,mdx}"
  if [[ "$CHECK_DOC_LINKS_REMOTE" != 0 ]]; then
    log "remote: curl unique http(s) targets (disable: CHECK_DOC_LINKS_REMOTE=0 or --local-only)"
  else
    log "remote: skipped (local paths only)"
  fi
  log "Markdown file(s) (${#DOC_FILES[@]}):"
  local md
  for md in "${DOC_FILES[@]}"; do
    case "$md" in
      "$REPO_ROOT"/*) log "  ${md#"$REPO_ROOT"/}" ;;
      *) log "  $md" ;;
    esac
  done

  local failures=0
  declare -a REMOTE_URLS=()

  log "phase 1/2: local file targets and Fern routes for [](url) / ![]() / <https://> (code fences skipped)"
  for md in "${DOC_FILES[@]}"; do
    if [[ ! -f "$md" ]]; then
      echo "check-docs-links: missing file: $md" >&2
      failures=1
      continue
    fi
    local target rc
    local _targets_output _targets_err
    _targets_err="$(mktemp)"
    if ! _targets_output="$(extract_targets "$md" 2>"$_targets_err")"; then
      echo "check-docs-links: malformed HTML comment in $md: $(tr '\n' ' ' <"$_targets_err" | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//')" >&2
      rm -f "$_targets_err"
      failures=1
      continue
    fi
    rm -f "$_targets_err"
    local line_no
    while IFS=$'\t' read -r line_no target || [[ -n "${target:-}" ]]; do
      [[ -z "$target" ]] && continue
      if check_local_ref "$md" "$line_no" "$target"; then
        rc=0
      else
        rc=$?
      fi
      if [[ "$rc" -eq 0 ]]; then
        continue
      elif [[ "$rc" -eq 2 ]]; then
        REMOTE_URLS+=("$target")
      else
        failures=1
      fi
    done <<<"$_targets_output"
  done

  if [[ "$failures" -ne 0 ]]; then
    log "phase 1 failed"
    return 1
  fi
  log "phase 1 OK (local paths and Fern routes resolve)"

  local _n_raw _deduped _unique _i _u url
  _n_raw="${#REMOTE_URLS[@]}"
  _deduped=""
  if [[ ${#REMOTE_URLS[@]} -gt 0 ]]; then
    _deduped="$(printf '%s\n' "${REMOTE_URLS[@]}" | LC_ALL=C sort -u)"
    _unique="$(printf '%s\n' "${REMOTE_URLS[@]}" | LC_ALL=C sort -u | grep -c . || true)"
  else
    _unique=0
  fi
  log "http(s): ${_n_raw} reference(s) -> ${_unique} unique URL(s)"
  if [[ -n "$_deduped" ]]; then
    log "unique http(s) URL(s) (alphabetically):"
    while IFS= read -r _u || [[ -n "${_u:-}" ]]; do
      [[ -z "${_u:-}" ]] && continue
      log "  ${_u}"
    done <<<"$_deduped"
  fi

  if [[ "$CHECK_DOC_LINKS_REMOTE" != 0 ]]; then
    if [[ -n "$_deduped" ]]; then
      local _probe_list="" _skip_count=0 _probe_n=0
      while IFS= read -r url || [[ -n "${url:-}" ]]; do
        [[ -z "${url:-}" ]] && continue
        if url_should_skip_remote_probe "$url"; then
          log "  skipped (ignore list): ${url}"
          _skip_count=$((_skip_count + 1))
        else
          _probe_list+="${url}"$'\n'
        fi
      done <<<"$_deduped"
      _probe_n="$(printf '%s\n' "$_probe_list" | grep -c . || true)"
      log "phase 2/2: curl ${_probe_n} URL(s), ${_skip_count} skipped (GET, -L, fail 4xx/5xx)"
      _i=0
      while IFS= read -r url || [[ -n "${url:-}" ]]; do
        [[ -z "${url:-}" ]] && continue
        _i=$((_i + 1))
        if [[ "$VERBOSE" -eq 1 ]]; then
          log "  [${_i}/${_probe_n}] ${url}"
        fi
        if ! check_remote_url "$url"; then
          failures=1
        fi
      done <<<"$_probe_list"
    else
      log "phase 2/2: no http(s) links"
    fi
  else
    if [[ -n "$_deduped" ]]; then
      log "phase 2/2: skipped ${_unique} URL(s) (local-only)"
    else
      log "phase 2/2: skipped (no http(s) links)"
    fi
  fi

  if [[ "$failures" -ne 0 ]]; then
    log "phase 2 failed"
    return 1
  fi
  log "summary: ${#DOC_FILES[@]} file(s), local OK$(
    [[ "$CHECK_DOC_LINKS_REMOTE" != 0 ]] && [[ ${_unique:-0} -gt 0 ]] && printf ', %s remote OK' "${_unique}"
  )$(
    [[ "$CHECK_DOC_LINKS_REMOTE" == 0 ]] && [[ ${_unique:-0} -gt 0 ]] && printf ' (%s remote not checked)' "${_unique}"
  )"
  log "done."
}

run_links_check
