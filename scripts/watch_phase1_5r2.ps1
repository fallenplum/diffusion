param(
    [string]$Root = 'D:\develop\diffusion',
    [int]$IntervalSeconds = 60
)

$ErrorActionPreference = 'Stop'
$python = 'D:\Anaconda\envs\tslib\python.exe'
$out = Join-Path $Root 'outputs\phase1_5r2_functional_repair'
$statePath = Join-Path $out 'run_state.json'
$lockPath = Join-Path $out 'run.lock'
$logPath = Join-Path $out 'logs\watcher.log'
New-Item -ItemType Directory -Force -Path (Split-Path $logPath) | Out-Null

function Write-WatchLog([string]$Message) {
    Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) $Message"
}

function Get-LockOwner {
    if (-not (Test-Path -LiteralPath $lockPath)) { return 0 }
    $text = Get-Content -LiteralPath $lockPath -Raw
    $match = [regex]::Match($text, 'pid=(\d+)')
    if (-not $match.Success) { return 0 }
    return [int]$match.Groups[1].Value
}

Write-WatchLog 'watcher started'
while ($true) {
    try {
        if (-not (Test-Path -LiteralPath $statePath)) {
            Write-WatchLog 'run_state.json missing; waiting'
            Start-Sleep -Seconds $IntervalSeconds
            continue
        }
        $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        $owner = Get-LockOwner
        $ownerAlive = $false
        if ($owner -gt 0) {
            $ownerAlive = [bool](Get-Process -Id $owner -ErrorAction SilentlyContinue)
        }

        if ($state.status -eq 'COMPLETE' -and -not $ownerAlive) {
            $validation = Join-Path $out 'phase1_5r2_validation.json'
            if (-not (Test-Path -LiteralPath $validation)) {
                Write-WatchLog 'run complete and lock released; starting final audit'
                & $python (Join-Path $Root 'scripts\finalize_phase1_5r2_audit.py') --root $Root *>> $logPath
                if ($LASTEXITCODE -ne 0) {
                    Write-WatchLog "final audit exited with code $LASTEXITCODE"
                    exit $LASTEXITCODE
                }
            }
            Write-WatchLog 'watcher finished after COMPLETE'
            break
        }
        if ($state.status -eq 'FAILED') {
            Write-WatchLog 'run state is FAILED; stopping watcher'
            break
        }
        if ($state.status -eq 'RUNNING' -and -not $ownerAlive) {
            Write-WatchLog "stale/missing runner owner=$owner; starting resumable runner"
            $stdout = Join-Path $out 'logs\runner_watch_stdout.log'
            $stderr = Join-Path $out 'logs\runner_watch_stderr.log'
            Start-Process -FilePath $python -ArgumentList @((Join-Path $Root 'scripts\run_phase1_5r2.py'), '--root', $Root, '--resume') -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr | Out-Null
        }
        Start-Sleep -Seconds $IntervalSeconds
    } catch {
        Write-WatchLog "watcher error: $($_.Exception.Message)"
        Start-Sleep -Seconds $IntervalSeconds
    }
}
