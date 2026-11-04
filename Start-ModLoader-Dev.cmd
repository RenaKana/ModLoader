@echo off
setlocal
cd /d "%~dp0"
if not defined MODLOADER_DATA_DIR set "MODLOADER_DATA_DIR=%~dp0data"
set "MODLOADER_DEV_URL="

if not exist "node_modules\electron\dist\electron.exe" (
  echo Electron is missing. Please run npm ci in this folder first.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Python environment is missing. See docs/development.md.
  pause
  exit /b 1
)

if /i "%~1"=="--build" goto build
goto launch

:build
where npm.cmd >nul 2>&1
if errorlevel 1 (
  echo npm is missing. Please install Node.js and open this script again.
  pause
  exit /b 1
)

echo Building the latest ModLoader development version...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\build-cpu-sensors.ps1
if errorlevel 1 (
  echo CPU sensor helper build failed. See docs/development.md.
  pause
  exit /b 1
)
call npm.cmd run build
if errorlevel 1 (
  echo.
  echo Build failed. ModLoader was not started. See the error above and docs/development.md.
  pause
  exit /b 1
)

:launch
if not exist "dist\index.html" (
  echo The interface has not been built. Run Start-ModLoader-Dev.cmd --build first.
  pause
  exit /b 1
)
start "" "node_modules\electron\dist\electron.exe" .
exit /b %errorlevel%
