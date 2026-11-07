@echo off
cd /d "%~dp0"
call agentlab\Scripts\activate
streamlit run app.py
