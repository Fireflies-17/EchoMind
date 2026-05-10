param(
  [Parameter(Mandatory = $true)]
  [string]$InputVideo,
  # Optional override. By default the run id is the input file name without extension.
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
