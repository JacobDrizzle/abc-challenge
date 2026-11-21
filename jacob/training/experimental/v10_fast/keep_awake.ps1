# Keeps Windows from sleeping while the v10_fast screen runs (the PC sleeps after 10 idle minutes). As
# experimental\v10_loo\keep_awake.ps1: it changes no power settings, it holds a system-required request
# (SetThreadExecutionState), which Windows drops when this script exits. The display can still turn off, and it
# does not stop a sleep started by hand.
#
#   -WatchPid <pid>   hold the request until that process exits (for a run that is already going)
#   no argument       run run_screen.cmd itself and hold the request until it finishes (also how to resume)
#
# powershell -NoProfile -ExecutionPolicy Bypass -File experimental\v10_fast\keep_awake.ps1 [-WatchPid <pid>]
param([int]$WatchPid = 0)

Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);
'@
$ES_CONTINUOUS = [uint32]'0x80000000'
$ES_SYSTEM_REQUIRED = [uint32]'0x00000001'
[void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
try {
    if ($WatchPid) {
        Wait-Process -Id $WatchPid -ErrorAction SilentlyContinue
    } else {
        & cmd.exe /c "`"$PSScriptRoot\run_screen.cmd`""
    }
} finally {
    [void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS)
}
