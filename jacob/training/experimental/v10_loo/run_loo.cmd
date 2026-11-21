@echo off
rem Leave-one-out ablation of k8_combined: screen (seed 42, control + 8 variants), then confirm (seeds 43, 44).
rem See README.md in this folder. Re-running this file resumes: finished seeds are reused.
cd /d <repo>\jacob\training
rem Fresh venv from the root requirements.txt (6 Nov). The ABC_local_test venv this first used was partly deleted
rem at 23:19 on 6 Nov, which stopped the screen after the control.
set PY=<venv>\Scripts\python.exe
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo [%date% %time%] screen start>> outputs\v10_loo_screen_run.log
"%PY%" -m best.run --stage screen --suite experimental/v10_loo/configs/screen.json --output outputs/v10_loo >> outputs\v10_loo_screen_run.log 2>&1
if errorlevel 1 (
  echo [%date% %time%] screen FAILED>> outputs\v10_loo_screen_run.log
  exit /b 1
)
echo [%date% %time%] screen done>> outputs\v10_loo_screen_run.log

echo [%date% %time%] confirm start>> outputs\v10_loo_confirm_run.log
"%PY%" -m best.confirm --plan experimental/v10_loo/configs/confirm.json --output outputs/v10_loo_confirm >> outputs\v10_loo_confirm_run.log 2>&1
if errorlevel 1 (
  echo [%date% %time%] confirm FAILED>> outputs\v10_loo_confirm_run.log
  exit /b 1
)
echo [%date% %time%] confirm done>> outputs\v10_loo_confirm_run.log
