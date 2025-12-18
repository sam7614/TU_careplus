@echo off
echo 🎓 동명대학교 생존분석 기반 위기 학생 관리 시스템 - 프로젝트 복원
echo ================================================================

echo 📋 시스템 요구사항 확인 중...

REM Python 설치 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python이 설치되어 있지 않습니다.
    echo 💡 https://www.python.org/downloads/ 에서 Python 3.8 이상을 설치하세요.
    pause
    exit /b 1
)

echo ✅ Python이 설치되어 있습니다.

REM pip 업그레이드
echo 📦 pip 업그레이드 중...
python -m pip install --upgrade pip

REM 필요한 패키지 설치
echo 📦 필요한 패키지들 설치 중...
pip install -r requirements.txt

REM 데이터 파일 확인
if not exist "students_sample.csv" (
    echo ❌ students_sample.csv 파일이 없습니다.
    echo 💡 샘플 데이터 파일을 확인하세요.
    pause
    exit /b 1
)

echo ✅ 데이터 파일 확인 완료

echo 🚀 프로젝트 복원 완료!
echo.
echo 💻 애플리케이션 실행 방법:
echo    1. streamlit run app.py
echo    2. 또는 run_app.bat 실행
echo.
echo 🌐 브라우저에서 http://localhost:8501 로 접속하세요.

pause