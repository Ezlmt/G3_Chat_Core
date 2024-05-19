@echo off
cd /d %~dp0

:loop
call .venv\Scripts\activate

call python ASR_setup.py

timeout /t 10
goto :loop