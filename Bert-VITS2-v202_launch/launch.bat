:loop
call venv\python.exe server_fastapi.py
timeout /t 10
goto :loop