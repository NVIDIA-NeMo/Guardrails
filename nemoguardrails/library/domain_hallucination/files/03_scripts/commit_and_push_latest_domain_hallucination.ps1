$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (git -C $scriptDir rev-parse --show-toplevel).Trim()
Set-Location $repoRoot

Write-Host "Current branch:"
git branch --show-current

Write-Host "Remote:"
git remote -v

$files = @(
  "pyproject.toml",
  "nemoguardrails/library/domain_hallucination",
  "tests/library/domain_hallucination"
)

Write-Host "Staging latest adapted domain_hallucination files..."
git add -- $files

Write-Host "Checking staged diff for accidental API keys..."
$secretDiff = git diff --cached --unified=0 | Select-String -Pattern "sk-[A-Za-z0-9]{20,}|api_key:\s*sk-|Authorization:\s*Bearer\s+sk-" -CaseSensitive
if ($secretDiff) {
  Write-Host "Potential secret found in staged diff. Aborting commit."
  $secretDiff
  exit 1
}

Write-Host "Staged files:"
git diff --cached --name-only

git commit -m "Add async domain hallucination verification and eval harness"

$currentBranch = (git branch --show-current).Trim()
if (-not $currentBranch) {
  throw "Unable to determine current branch."
}

Write-Host "Pushing to origin $currentBranch..."
git push origin $currentBranch
