@echo off
title RAG Chatbot - Auto Setup
echo ============================================
echo  RAG CHATBOT v7 - AUTO SETUP
echo  Just wait. Do NOT close. Do NOT press Ctrl+C.
echo ============================================
echo.
python --version
if errorlevel 1 (
  echo.
  echo [ERROR] Python not found! Install Python from python.org
  echo Tick ADD PYTHON TO PATH during install, then run setup.bat again.
  pause
  exit /b 1
)
echo.
if not exist venv (
  echo [1/4] Creating venv folder... please wait...
  python -m venv venv
  if errorlevel 1 (
    echo [ERROR] venv failed. Try: python -m pip install virtualenv
    pause
    exit /b 1
  )
  echo [OK] venv created.
) else (
  echo [1/4] venv already exists - OK.
)
echo.
echo [2/4] Upgrading pip...
venv\Scripts\python.exe -m pip install --upgrade pip
echo.
echo [3/4] Installing all packages - 3 to 5 min - DO NOT CLOSE...
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo [ERROR] Install failed. Check internet, then run setup.bat again.
  pause
  exit /b 1
)
echo.
echo [4/4] Checking...
venv\Scripts\python.exe -c "from langchain_huggingface import HuggingFaceEmbeddings; print('OK - embeddings work')"
echo.
echo ============================================
echo  DONE! Opening app at http://localhost:8501
echo ============================================
venv\Scripts\python.exe -m streamlit run app.py
pause
