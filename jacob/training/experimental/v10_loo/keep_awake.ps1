# Keeps Windows from sleeping while the leave-one-out suite runs. It changes no power settings: it holds a
# system-required request (SetThreadExecutionState), which Windows drops when this script exits. The display can still
# turn off.
#
#   -WatchPid <pid>   hold the request until that process exits (used on 6 Nov for an already started run)
#   no argument       run run_loo.cmd itself and hold the request until it finishes (use this to resume)
#
# powershell -NoProfile -ExecutionPolicy Bypass -File experimental\v10_loo\keep_awake.ps1 [-WatchPid <pid>]
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
        & cmd.exe /c "`"$PSScriptRoot\run_loo.cmd`""
    }
} finally {
    [void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS)
}
