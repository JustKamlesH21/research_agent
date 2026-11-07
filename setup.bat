@echo off
cd /d "%~dp0"
if not exist agentlab python -m venv agentlab
call agentlab\Scripts\activate
pip install -r requirements.txt
if not exist .env copy .env.example .env
python test_offline.py
python tools.py
echo.
echo Now open .env in Notepad and paste your API key, then run run_ui.bat
pause
