param(
    [string]$ConfigPath = "$HOME\.codex\config.toml"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw "Codex config not found: $ConfigPath"
}

$content = Get-Content -LiteralPath $ConfigPath -Raw
$providerMatch = [regex]::Match($content, '(?m)^model_provider\s*=\s*"([^"]*)"')

if (-not $providerMatch.Success) {
    throw "model_provider entry not found in $ConfigPath"
}

$currentProvider = $providerMatch.Groups[1].Value

switch ($currentProvider) {
    "packycode" {
        $targetProvider = "local_gpt54"
    }
    "local_gpt54" {
        $targetProvider = "packycode"
    }
    default {
        throw "Unsupported current provider '$currentProvider'. Supported values: packycode, local_gpt54"
    }
}

& "$PSScriptRoot\switch-codex-provider.ps1" -Provider $targetProvider
