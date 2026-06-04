param(
  [Parameter(Mandatory=$true)]
  [string]$Provider,

  [Parameter(Mandatory=$true)]
  [string]$ConfigFile,

  [Parameter(Mandatory=$true)]
  [string]$ApiKeyEnvVar
)

$ErrorActionPreference = "Stop"

Set-Location "E:\123\Guardrails\nemoguardrails\library\domain_hallucination\files"

if (-not [Environment]::GetEnvironmentVariable($ApiKeyEnvVar, "Process")) {
  Write-Host "$ApiKeyEnvVar is not set. Expert review requires the configured LLM."
  exit 1
}

$prefix = $Provider.ToLower()

$runs = @(
  @{
    Name = "$Provider Expert S1 cached full"
    Args = @("--experiments", "A", "--datasets", "eval_dataset.json", "--domain-verification-level", "full", "--enable-domain-expert-review", "--nemo-config", $ConfigFile, "--skip-baseline", "--output", "$($prefix)_expert_S1_cached_full_eval58.json")
    Output = "$($prefix)_expert_S1_cached_full_eval58.json"
  },
  @{
    Name = "$Provider Expert S2 cached full skip DNS-failed secondary checks"
    Args = @("--experiments", "A", "--datasets", "eval_dataset.json", "--domain-verification-level", "full", "--skip-secondary-checks-on-dns-failure", "--enable-domain-expert-review", "--nemo-config", $ConfigFile, "--skip-baseline", "--output", "$($prefix)_expert_S2_cached_full_skip_dnsfail_eval58.json")
    Output = "$($prefix)_expert_S2_cached_full_skip_dnsfail_eval58.json"
  },
  @{
    Name = "$Provider Expert S3 cached full skip DNS-failed secondary checks no WHOIS"
    Args = @("--experiments", "A", "--datasets", "eval_dataset.json", "--domain-verification-level", "full", "--skip-secondary-checks-on-dns-failure", "--disable-domain-whois", "--enable-domain-expert-review", "--nemo-config", $ConfigFile, "--skip-baseline", "--output", "$($prefix)_expert_S3_cached_full_skip_dnsfail_no_whois_eval58.json")
    Output = "$($prefix)_expert_S3_cached_full_skip_dnsfail_no_whois_eval58.json"
  },
  @{
    Name = "$Provider Expert S4 http skip DNS-failed secondary checks"
    Args = @("--experiments", "A", "--datasets", "eval_dataset.json", "--domain-verification-level", "http", "--skip-secondary-checks-on-dns-failure", "--enable-domain-expert-review", "--nemo-config", $ConfigFile, "--skip-baseline", "--output", "$($prefix)_expert_S4_http_skip_dnsfail_eval58.json")
    Output = "$($prefix)_expert_S4_http_skip_dnsfail_eval58.json"
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
  "$($prefix)_expert_S1_cached_full_eval58.json" `
  "$($prefix)_expert_S2_cached_full_skip_dnsfail_eval58.json" `
  "$($prefix)_expert_S3_cached_full_skip_dnsfail_no_whois_eval58.json" `
  "$($prefix)_expert_S4_http_skip_dnsfail_eval58.json"

