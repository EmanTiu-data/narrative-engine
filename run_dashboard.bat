@echo off
cd /d "%~dp0"
echo Starting Narrative Intelligence Engine Dashboard...
python -m streamlit run app/dashboard.py --server.port 8503
pause
