@echo off
rem Train v5, test it, train v6, test it, on the group-a-v2 export in <repo>\data\h5 (the README has the details).
rem Saved epochs are kept: run this again after an interruption and every stage resumes or verifies what it has.
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" set "ABC_PYTHON=%~dp0.venv\Scripts\python.exe"
if not defined ABC_PYTHON (
  echo No Python found. Create jacob\training\.venv from requirements-training.txt, or set ABC_PYTHON
  echo to an interpreter that has them, e.g. set ABC_PYTHON=<local_test>\.venv\Scripts\python.exe
  goto :failed
)
set V5=outputs\v5_group_a_v2
set V6=outputs\v6_group_a_v2
"%ABC_PYTHON%" -B -u desktop_next.py --stage all --output %V5% --suite configs\v5_run.json --base-config configs\tcnn_v5.json
if errorlevel 1 goto :failed
"%ABC_PYTHON%" -B -u evaluate_tcnn.py --checkpoint %V5%\selected_model.pt --selection %V5%\selection.json --damped-selection %V5%\damped_baseline\selection.json --split test --device cuda --output %V5%\test
if errorlevel 1 goto :failed
"%ABC_PYTHON%" -B -u -m experimental.v6_gravity_mirror.run --stage all --output %V6%
if errorlevel 1 goto :failed
"%ABC_PYTHON%" -B -u -m experimental.v6_gravity_mirror.evaluate --checkpoint %V6%\selected_model.pt --selection %V6%\selection.json --damped-selection %V6%\damped_baseline\selection.json --split test --device cuda --output %V6%\test
if errorlevel 1 goto :failed
echo Done. Test reports: %V5%\test\report.md and %V6%\test\report.md
pause
exit /b 0
:failed
echo Run stopped. Read the error above. Saved epochs are kept; run this again to resume.
pause
exit /b 1
