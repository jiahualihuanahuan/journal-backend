@echo off
REM Windows / Anaconda: conda activate journal 后运行 scripts\run.bat
cd /d "%~dp0.."
if "%HOST%"=="" set HOST=127.0.0.1
if "%PORT%"=="" set PORT=8765
uvicorn app.main:get_app --factory --host %HOST% --port %PORT%
