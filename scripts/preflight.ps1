# Memory check before a long LLM run on the Windows laptop (docs/design/llm_setup.md).
# Usage, from any terminal:  powershell -ExecutionPolicy Bypass -File scripts/preflight.ps1
# Prints PASS or FAIL per check with the fix; exits 1 if any check fails.

$failed = $false

function Report([bool]$ok, [string]$what, [string]$fix) {
    # Write-Host, not Write-Output: a function's output stream would become its return value.
    if ($ok) {
        Write-Host "PASS  $what"
    } else {
        Write-Host "FAIL  $what"
        Write-Host "      fix: $fix"
    }
    return -not $ok
}

$os = Get-CimInstance Win32_OperatingSystem
$uptime = (Get-Date) - $os.LastBootUpTime
Write-Output ("Booted {0} ({1:N1} days ago)" -f $os.LastBootUpTime, $uptime.TotalDays)

# 1. Free memory: the server crashed with about 3.6 GB free; long runs need headroom.
$availableGb = (Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue / 1024
$failed = (Report ($availableGb -ge 12) ("free RAM {0:N1} GB (need 12+)" -f $availableGb) `
    "close big apps, or Restart Windows (Start > Power > Restart, not Shut down)") -or $failed

# 2. Shared GPU memory held by one process (the Desktop Window Manager leak seen on Nov 9).
$shared = (Get-Counter '\GPU Process Memory(*)\Shared Usage').CounterSamples |
    Sort-Object CookedValue -Descending | Select-Object -First 1
$sharedGb = $shared.CookedValue / 1GB
$failed = (Report ($sharedGb -lt 1) ("largest shared GPU memory {0:N2} GB (need under 1)" -f $sharedGb) `
    "Restart Windows, or sign out and back in") -or $failed

# 3. Zombie processes: exited processes Windows never released, about 90 KB each (Nov 14).
$objects = (Get-Counter '\Objects\Processes').CounterSamples[0].CookedValue
$live = (Get-Process).Count
$zombies = $objects - $live
$failed = (Report ($zombies -lt 5000) ("{0:N0} leftover processes (need under 5,000)" -f $zombies) `
    "Restart Windows; then fix what keeps starting processes (check 4)") -or $failed

# 4. Something polling docker.exe (VS Code's Docker extensions) creates the zombies.
$seen = @{}
$end = (Get-Date).AddSeconds(10)
while ((Get-Date) -lt $end) {
    Get-Process -Name docker -ErrorAction SilentlyContinue | ForEach-Object { $seen[$_.Id] = 1 }
    Start-Sleep -Milliseconds 100
}
$failed = (Report ($seen.Count -eq 0) ("docker.exe started {0} times in 10 s (need 0)" -f $seen.Count) `
    "VS Code Extensions panel: Disable Docker, Docker DX, Container Tools, Dev Containers") -or $failed

if ($failed) {
    Write-Output "Preflight FAILED: fix the items above before a long run."
    exit 1
}
Write-Output "Preflight passed."
