@echo off
echo 🎓 동명대학교 위기 학생 관리 시스템
echo ================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python이 설치되어 있지 않습니다.
    echo 💡 https://www.python.org/downloads/ 에서 Python을 다운로드하세요.
    pause
    exit /b 1
)

echo ✅ Python이 설치되어 있습니다.

REM Install requirements
echo 📦 필요한 패키지를 설치하는 중...
pip install -r requirements.txt

REM Check if data file exists
if not exist "care_student.csv" (
    if not exist "students_sample.csv" (
        echo ❌ 데이터 파일을 찾을 수 없습니다.
        echo 💡 care_student.csv 또는 students_sample.csv 파일을 준비하세요.
        pause
        exit /b 1
    ) else (
        echo 📋 샘플 데이터 파일을 사용합니다.
        copy students_sample.csv care_student.csv
    )
)

echo 🚀 애플리케이션을 시작합니다...
echo 💻 브라우저에서 http://localhost:8501 로 접속하세요.
echo 🛑 종료하려면 Ctrl+C를 누르세요.
echo.

streamlit run app.py

pause