$ErrorActionPreference = "Stop"

Set-Location "E:\123\Guardrails"

$env:HTTP_PROXY = "http://127.0.0.1:7897"
$env:HTTPS_PROXY = "http://127.0.0.1:7897"

Write-Host "Current branch:"
git branch --show-current

Write-Host "Remote:"
git remote -v

$files = @(
  "pyproject.toml",
  "nemoguardrails/library/domain_hallucination/CHANGELOG.md",
  "nemoguardrails/library/domain_hallucination/README.md",
  "nemoguardrails/library/domain_hallucination/__init__.py",
  "nemoguardrails/library/domain_hallucination/actions.py",
  "nemoguardrails/library/domain_hallucination/checkers.py",
  "nemoguardrails/library/domain_hallucination/config.py",
  "nemoguardrails/library/domain_hallucination/decision.py",
  "nemoguardrails/library/domain_hallucination/example_config.json",
  "nemoguardrails/library/domain_hallucination/extractors.py",
  "nemoguardrails/library/domain_hallucination/kb.py",
  "nemoguardrails/library/domain_hallucination/scoring.py",
  "nemoguardrails/library/domain_hallucination/semantic.py",
  "nemoguardrails/library/domain_hallucination/verification.py",
  "nemoguardrails/library/domain_hallucination/files/.gitignore",
  "nemoguardrails/library/domain_hallucination/files/EVAL_README.md",
  "nemoguardrails/library/domain_hallucination/files/comparison_report.md",
  "nemoguardrails/library/domain_hallucination/files/deepseek_config.yml",
  "nemoguardrails/library/domain_hallucination/files/eval_dataset.json",
  "nemoguardrails/library/domain_hallucination/files/eval_runner.py",
  "nemoguardrails/library/domain_hallucination/files/expanded_dataset.json",
  "nemoguardrails/library/domain_hallucination/files/experiment_manifest_g1.json",
  "nemoguardrails/library/domain_hallucination/files/expert_optimization_plan_A_eval58.md",
  "nemoguardrails/library/domain_hallucination/files/full_dataset.json",
  "nemoguardrails/library/domain_hallucination/files/generate_dataset.py",
  "nemoguardrails/library/domain_hallucination/files/generate_report.py",
  "nemoguardrails/library/domain_hallucination/files/optimization_plan_A_eval58.md",
  "nemoguardrails/library/domain_hallucination/files/run_A_eval58_expert_sweep.ps1",
  "nemoguardrails/library/domain_hallucination/files/run_A_eval58_expert_sweep_resume.ps1",
  "nemoguardrails/library/domain_hallucination/files/run_A_eval58_optimization_sweep.ps1",
  "nemoguardrails/library/domain_hallucination/files/run_ablation_experiments.py",
  "nemoguardrails/library/domain_hallucination/files/summarize_optimization_results.py",
  "nemoguardrails/library/domain_hallucination/files/commit_and_push_latest_domain_hallucination.ps1"
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

Write-Host "Pushing to origin develop through proxy 7897..."
git push origin develop

