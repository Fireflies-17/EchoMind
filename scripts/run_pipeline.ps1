param(
  [Parameter(Mandatory = $true)]
  [string]$InputVideo,
  [string]$RunId = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $repo "src"

if ($RunId) {
  python -m video_kb.cli run --input $InputVideo --run-id $RunId
} else {
  python -m video_kb.cli run --input $InputVideo
}

