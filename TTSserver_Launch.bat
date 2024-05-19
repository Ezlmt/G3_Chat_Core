@echo off
cd /d %~dp0

:loop
call .venv\Scripts\activate

call python TTS_setup.py

timeout /t 10
goto :loop