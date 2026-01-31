@echo off
REM DWTS 粉丝投票 MCMC 反推 - 一键启动脚本
REM 使用方法: 双击运行或在命令行执行

echo ============================================================
echo MCM 2026 Problem C - Q1
echo MCMC 粉丝投票反推模型
echo ============================================================
echo.

REM 切换到脚本所在目录
cd /d "%~dp0"

REM 检查 Python 环境
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请确保已安装并添加到 PATH
    pause
    exit /b 1
)

REM 检查依赖
python -c "import numpy, pandas, scipy, tqdm" >nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] 缺少依赖包，正在安装...
    pip install numpy pandas scipy tqdm matplotlib
)

REM 运行主程序
echo 开始运行...
echo.

python q1_mcmc/main.py %*

echo.
echo ============================================================
echo 运行完成！结果已保存到 outputs/q1_mcmc/ 目录
echo ============================================================

pause
