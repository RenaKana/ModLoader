@echo off
setlocal
cd /d "%~dp0"
set "MODLOADER_DATA_DIR=%~dp0data"
if exist "%~dp0Start-ModLoader.local.cmd" call "%~dp0Start-ModLoader.local.cmd"
if exist "release\win-unpacked\ModLoader.exe" (
  start "" "release\win-unpacked\ModLoader.exe"
  exit /b
)
if not exist "node_modules\electron\dist\electron.exe" (
  echo Please run npm install and install requirements.txt into .venv first.
  pause
  exit /b 1
)
start "" "node_modules\electron\dist\electron.exe" .
