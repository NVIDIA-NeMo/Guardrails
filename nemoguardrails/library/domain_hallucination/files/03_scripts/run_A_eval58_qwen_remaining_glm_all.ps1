$ErrorActionPreference = "Stop"

Set-Location "E:\123\Guardrails\nemoguardrails\library\domain_hallucination\files"

if (-not $env:QWEN_API_KEY) {
  Write-Host "QWEN_API_KEY is not set."
  exit 1
}
if (-not $env:GLM_API_KEY) {
  Write-Host "GLM_API_KEY is not set."
  exit 1
}

$runs = @(
  @{
    Name = "Qwen Expert S3 remaining cached full skip DNS-failed secondary checks no WHOIS"
    Args = @("--experiments", "A", "--datasets", "eval_dataset.json", "--domain-verification-level", "full", "--skip-secondary-checks-on-dns-failure", "--disable-domain-whois", "--enable-domain-expert-review", "--nemo-config", "qwen_config.yml", "--skip-baseline", "--output", "qwen_expert_S3_cached_full_skip_dnsfail_no_whois_eval58_resume.json")
    Output = "qwen_expert_S3_cached_full_skip_dnsfail_no_whois_eval58_resume.json"
  },
  @{
    Name = "Qwen Expert S4 remaining http skip DNS-failed secondary checks"
    Args = @("--experiments", "A", "--datasets", "eval_dataset.json", "--domain-verification-level", "http", "--skip-secondary-checks-on-dns-failure", "--enable-domain-expert-review", "--nemo-config", "qwen_config.yml", "--skip-baseline", "--output", "qwen_expert_S4_http_skip_dnsfail_eval58_resume.json")
    Output = "qwen_expert_S4_http_skip_dnsfail_eval58_resume.json"
  },
  @{
    Name = "GLM Expert S1 cached full"
    Args = @("--experiments", "A", "--datasets", "eval_dataset.json", "--domain-verification-level", "full", "--enable-domain-expert-review", "--nemo-config", "glm_config.yml", "--skip-baseline", "--output", "glm_expert_S1_cached_full_eval58.json")
    Output = "glm_expert_S1_cached_full_eval58.json"
  },
  @{
    Name = "GLM Expert S2 cached full skip DNS-failed secondary checks"
    Args = @("--experiments", "A", "--datasets", "eval_dataset.json", "--domain-verification-level", "full", "--skip-secondary-checks-on-dns-failure", "--enable-domain-expert-review", "--nemo-config", "glm_config.yml", "--skip-baseline", "--output", "glm_expert_S2_cached_full_skip_dnsfail_eval58.json")
    Output = "glm_expert_S2_cached_full_skip_dnsfail_eval58.json"
  },
  @{
    Name = "GLM Expert S3 cached full skip DNS-failed secondary checks no WHOIS"
    Args = @("--experiments", "A", "--datasets", "eval_dataset.json", "--domain-verification-level", "full", "--skip-secondary-checks-on-dns-failure", "--disable-domain-whois", "--enable-domain-expert-review", "--nemo-config", "glm_config.yml", "--skip-baseline", "--output", "glm_expert_S3_cached_full_skip_dnsfail_no_whois_eval58.json")
    Output = "glm_expert_S3_cached_full_skip_dnsfail_no_whois_eval58.json"
  },
  @{
    Name = "GLM Expert S4 http skip DNS-failed secondary checks"
    Args = @("--experiments", "A", "--datasets", "eval_dataset.json", "--domain-verification-level", "http", "--skip-secondary-checks-on-dns-failure", "--enable-domain-expert-review", "--nemo-config", "glm_config.yml", "--skip-baseline", "--output", "glm_expert_S4_http_skip_dnsfail_eval58.json")
    Output = "glm_expert_S4_http_skip_dnsfail_eval58.json"
  }
)

foreach ($run in $runs) {
  if (Test-Path $run.Output) {
    Write-Host ""
    Write-Host "=== Skipping existing $($run.Output) ==="
    continue
  }

  Write-Host ""
  Write-Host "=== Running $($run.Name) ==="
  python .\run_ablation_experiments.py @($run.Args)
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  if (Test-Path $run.Output) {
    Copy-Item -LiteralPath $run.Output -Destination "$($run.Output).backup_$stamp"
  }
  if (Test-Path "$($run.Output).partial.json") {
    Copy-Item -LiteralPath "$($run.Output).partial.json" -Destination "$($run.Output).partial.json.backup_$stamp"
  }
}

python .\summarize_model_expert_comparison.py
