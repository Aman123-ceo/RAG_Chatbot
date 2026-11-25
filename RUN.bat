@echo off
title RAG Chatbot - Run
if not exist venv (
  echo No venv found - running setup first...
  call setup.bat
  exit /b 0
)
echo Starting app...
venv\Scripts\python.exe -m streamlit run app.py
pause
