$ErrorActionPreference = "Stop"

Set-Location "E:\123\Guardrails\nemoguardrails\library\domain_hallucination\files"

$required = @("DEEPSEEK_API_KEY", "QWEN_API_KEY", "GLM_API_KEY", "OPENROUTER_API_KEY")
foreach ($name in $required) {
  if (-not [Environment]::GetEnvironmentVariable($name, "Process")) {
    Write-Host "$name is not set."
    exit 1
  }
}

$models = @(
  @{ Prefix = "deepseek"; Label = "DeepSeek"; Config = "deepseek_config.yml" },
  @{ Prefix = "qwen"; Label = "Qwen"; Config = "qwen_config.yml" },
  @{ Prefix = "glm_air"; Label = "GLM-air"; Config = "glm_air_config.yml" },
  @{ Prefix = "openrouter_gpt41mini"; Label = "OpenRouter GPT-4.1-mini"; Config = "openrouter_gpt41mini_config.yml" }
)

$strategies = @(
  @{
    Id = "S1"
    Name = "cached full"
    Extra = @("--domain-verification-level", "full")
    Suffix = "cached_full"
  },
  @{
    Id = "S2"
    Name = "cached full skip DNS-failed secondary checks"
    Extra = @("--domain-verification-level", "full", "--skip-secondary-checks-on-dns-failure")
    Suffix = "cached_full_skip_dnsfail"
  },
  @{
    Id = "S3"
    Name = "cached full skip DNS-failed secondary checks no WHOIS"
    Extra = @("--domain-verification-level", "full", "--skip-secondary-checks-on-dns-failure", "--disable-domain-whois")
    Suffix = "cached_full_skip_dnsfail_no_whois"
  },
  @{
    Id = "S4"
    Name = "http skip DNS-failed secondary checks"
    Extra = @("--domain-verification-level", "http", "--skip-secondary-checks-on-dns-failure")
    Suffix = "http_skip_dnsfail"
  }
)

foreach ($model in $models) {
  foreach ($strategy in $strategies) {
    $output = "$($model.Prefix)_expert_$($strategy.Id)_$($strategy.Suffix)_full223.json"

    if (Test-Path $output) {
      Write-Host ""
      Write-Host "=== Skipping existing $output ==="
      continue
    }

    Write-Host ""
    Write-Host "=== Running $($model.Label) Expert $($strategy.Id): $($strategy.Name) on full_dataset.json ==="
    $args = @(
      "--experiments", "A",
      "--datasets", "full_dataset.json",
      "--enable-domain-expert-review",
      "--nemo-config", $model.Config,
      "--skip-baseline",
      "--resume-from-partial",
      "--output", $output
    ) + $strategy.Extra

    python .\run_ablation_experiments.py @args

    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    if (Test-Path $output) {
      Copy-Item -LiteralPath $output -Destination "$output.backup_$stamp"
    }
    if (Test-Path "$output.partial.json") {
      Copy-Item -LiteralPath "$output.partial.json" -Destination "$output.partial.json.backup_$stamp"
    }
  }
}
