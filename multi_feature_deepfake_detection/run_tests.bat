@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found at .venv\Scripts\python.exe
    exit /b 1
)

mkdir reports 2>nul
mkdir reports\execution_logs 2>nul

echo [1/3] Running Ruff lint checks...
.venv\Scripts\ruff.exe check models data training evaluation explainability app > reports\execution_logs\ruff.log 2>&1

echo [2/3] Running full pytest suite with coverage...
.venv\Scripts\pytest.exe tests -v --cov=models --cov=data --cov=training --cov=evaluation --cov=explainability --cov-report=html:reports/coverage --cov-report=json:reports/test_summary.json --cov-report=term > reports\execution_logs\pytest.log 2>&1

echo [3/3] Done. See reports\execution_logs\pytest.log and reports\coverage\index.html
endlocal
