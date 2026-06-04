$ErrorActionPreference = "Stop"

Set-Location "E:\123\Guardrails\nemoguardrails\library\domain_hallucination\files"

if (-not $env:DEEPSEEK_API_KEY) {
  Write-Host "DEEPSEEK_API_KEY is not set. Expert review requires the NeMo-configured LLM."
  exit 1
}

$runs = @(
  @{
    Name = "Expert S2 resumed cached full skip DNS-failed secondary checks"
    Args = @("--experiments", "A", "--datasets", "eval_dataset.json", "--domain-verification-level", "full", "--skip-secondary-checks-on-dns-failure", "--enable-domain-expert-review", "--skip-baseline", "--output", "expert_S2_cached_full_skip_dnsfail_eval58_resume.json")
    Output = "expert_S2_cached_full_skip_dnsfail_eval58_resume.json"
  },
  @{
    Name = "Expert S3 cached full skip DNS-failed secondary checks no WHOIS"
    Args = @("--experiments", "A", "--datasets", "eval_dataset.json", "--domain-verification-level", "full", "--skip-secondary-checks-on-dns-failure", "--disable-domain-whois", "--enable-domain-expert-review", "--skip-baseline", "--output", "expert_S3_cached_full_skip_dnsfail_no_whois_eval58_resume.json")
    Output = "expert_S3_cached_full_skip_dnsfail_no_whois_eval58_resume.json"
  },
  @{
    Name = "Expert S4 http skip DNS-failed secondary checks"
    Args = @("--experiments", "A", "--datasets", "eval_dataset.json", "--domain-verification-level", "http", "--skip-secondary-checks-on-dns-failure", "--enable-domain-expert-review", "--skip-baseline", "--output", "expert_S4_http_skip_dnsfail_eval58_resume.json")
    Output = "expert_S4_http_skip_dnsfail_eval58_resume.json"
  }
)

foreach ($run in $runs) {
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

python .\summarize_optimization_results.py `
  expert_S1_cached_full_eval58.json `
  expert_S2_cached_full_skip_dnsfail_eval58_resume.json `
  expert_S3_cached_full_skip_dnsfail_no_whois_eval58_resume.json `
  expert_S4_http_skip_dnsfail_eval58_resume.json
