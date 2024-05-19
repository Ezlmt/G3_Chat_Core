@echo off
cd /d %~dp0

:loop
call .venv\Scripts\activate

call python LLM_setup.py

timeout /t 10
goto :loop