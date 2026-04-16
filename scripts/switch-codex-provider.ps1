param(
    [Parameter(Mandatory = $true)]
    [string]$Provider,

    [string]$Model = "gpt-5.4",

    [string]$ConfigPath = "$HOME\.codex\config.toml"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw "Codex config not found: $ConfigPath"
}

$content = Get-Content -LiteralPath $ConfigPath -Raw

$providerPattern = '(?m)^model_provider\s*=\s*"[^"]*"'
$modelPattern = '(?m)^model\s*=\s*"[^"]*"'

if ($content -notmatch $providerPattern) {
    throw "model_provider entry not found in $ConfigPath"
}

if ($content -notmatch $modelPattern) {
    throw "model entry not found in $ConfigPath"
}

$updated = [regex]::Replace($content, $providerPattern, "model_provider = `"$Provider`"")
$updated = [regex]::Replace($updated, $modelPattern, "model = `"$Model`"")

if ($updated -eq $content) {
    Write-Host "No config change needed. Provider=$Provider Model=$Model"
    exit 0
}

Set-Content -LiteralPath $ConfigPath -Value $updated -NoNewline

Write-Host "Switched Codex provider to '$Provider' with model '$Model'."
Write-Host "Config: $ConfigPath"
Write-Host "Restart the Codex session in VS Code to apply the change."
