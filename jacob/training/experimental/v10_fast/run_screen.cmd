@echo off
rem v10_fast: seed-42 screen on the 800-user file with the fast forward (control, dropout_0, passes_2, jitter_1_5,
rem width_96). See README.md in this folder. Re-running this file resumes: finished variants are reused, and an
rem interrupted one continues from its last saved epoch. To run only some variants, add --variants <names> below.
cd /d <repo>\jacob\training
set PY=<venv>\Scripts\python.exe
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set DATA=..\..\data\h5\qualified_head_100hz_merged_20261101_140915.h5

echo [%date% %time%] screen start>> outputs\v10_fast_800_run.log
"%PY%" -m experimental.v10_fast.run --stage screen --data %DATA% --output outputs/v10_fast_800 >> outputs\v10_fast_800_run.log 2>&1
if errorlevel 1 (
  echo [%date% %time%] screen FAILED>> outputs\v10_fast_800_run.log
  exit /b 1
)
echo [%date% %time%] screen done>> outputs\v10_fast_800_run.log
