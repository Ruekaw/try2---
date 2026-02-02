@echo off
setlocal EnableExtensions

REM One-click launcher for q1_mcmc (Windows)

REM Use UTF-8 code page for cleaner output (best-effort)
chcp 65001 >nul

echo ============================================================
echo MCM 2026 Problem C - Q1
echo MCMC Fan Vote Inference (Q1)
echo ============================================================
echo.

REM Switch to workspace root (script directory)
cd /d "%~dp0"

REM Pick Python: prefer the conda env used in this workspace
set "PYTHON="
for %%P in (
    "D:\Anaconda\envs\mcm_env\python.exe"
    "D:\Anaconda3\envs\mcm_env\python.exe"
    "D:\Anaconda\python.exe"
    "D:\Anaconda3\python.exe"
) do (
    if not defined PYTHON if exist %%~fP set "PYTHON=%%~fP"
)

if not defined PYTHON (
    where python >nul 2>&1
    if %errorlevel% equ 0 set "PYTHON=python"
)

if not defined PYTHON (
    echo [ERROR] Python not found.
    echo Please install Python, add it to PATH, or edit run_q1_mcmc.bat to point to your conda env.
    pause
    exit /b 1
)

REM Check deps (install if missing)
"%PYTHON%" -c "import numpy, pandas, scipy, tqdm" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Missing packages detected, installing...
    "%PYTHON%" -m pip install numpy pandas scipy tqdm matplotlib
)

REM Default parallel: use CPU cores - 1 (minimum 1)
set "N_JOBS=%NUMBER_OF_PROCESSORS%"
set /a N_JOBS=%N_JOBS%-1
if %N_JOBS% LSS 1 set "N_JOBS=1"

echo Running with: -j %N_JOBS%  (override by passing -j N or --no-parallel)
echo.

set "EXPORT_ARGS="
set /p "EXPORT_CHOICE=Enable sample export (.npz)? [y/N]: "
if /I "%EXPORT_CHOICE%"=="y" (
    set /p "EXPORT_PATH=Export directory (blank=outputs\q1_mcmc): "
    if "%EXPORT_PATH%"=="" (
        set "EXPORT_ARGS=--export-samples"
    ) else (
        set "EXPORT_ARGS=--export-samples --samples-dir %EXPORT_PATH%"
    )
)

"%PYTHON%" q1_mcmc/main.py -j %N_JOBS% %EXPORT_ARGS% %*

echo.
echo ============================================================
echo Done. Outputs saved under outputs/q1_mcmc/
echo ============================================================

pause
