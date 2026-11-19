@echo off
REM One-click setup for VibeMatch AI. Double-click this file in File Explorer.
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe

echo.
echo === 1/4  Getting the latest code from GitHub ===
where git >nul 2>&1 || (echo Git is not installed or not on PATH. Install it from https://git-scm.com and run this again. & pause & exit /b 1)
git pull --ff-only origin main || (echo. & echo git pull failed - read the message above. & pause & exit /b 1)

echo.
echo === 2/4  Installing packages into .venv (first time takes a few minutes) ===
if not exist %PY% py -3 -m venv .venv
%PY% -m pip install --upgrade pip -q
%PY% -m pip install -r requirements.txt || (echo. & echo Package install failed - read the message above. & pause & exit /b 1)

echo.
echo === 3/4  Running the tests ===
%PY% -m pytest -q
if errorlevel 1 (echo. & echo Some tests failed - see above. Starting the app anyway.)

echo.
echo === 4/4  Starting the app at http://localhost:8501 ===
echo Keep this window open while you use the app. Close it to stop the app.
if not exist .env echo No .env file found: the app runs in OFFLINE mode. See README step 3 to add an API key.
start "" cmd /c "timeout /t 5 >nul & start http://localhost:8501"
%PY% -m streamlit run app.py --server.headless true
pause
