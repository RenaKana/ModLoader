@echo off
setlocal
cd /d "%~dp0"
set "MODLOADER_DATA_DIR=%~dp0data"
set "MODLOADER_DEV_URL="
if exist "%~dp0Start-ModLoader.local.cmd" call "%~dp0Start-ModLoader.local.cmd"
if not exist "release\win-unpacked\ModLoader.exe" (
  echo Packaged ModLoader is missing: release\win-unpacked\ModLoader.exe
  echo Run npm run package to build it, or use Start-ModLoader-Dev.cmd for source development.
  pause
  exit /b 1
)
start "" "release\win-unpacked\ModLoader.exe"
exit /b %errorlevel%
