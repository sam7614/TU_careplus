#!/bin/bash

echo "🎓 동명대학교 위기 학생 관리 시스템"
echo "================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ Python이 설치되어 있지 않습니다."
        echo "💡 https://www.python.org/downloads/ 에서 Python을 다운로드하세요."
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "✅ Python이 설치되어 있습니다."

# Install requirements
echo "📦 필요한 패키지를 설치하는 중..."
$PYTHON_CMD -m pip install -r requirements.txt

# Check if data file exists
if [ ! -f "care_student.csv" ]; then
    if [ ! -f "students_sample.csv" ]; then
        echo "❌ 데이터 파일을 찾을 수 없습니다."
        echo "💡 care_student.csv 또는 students_sample.csv 파일을 준비하세요."
        exit 1
    else
        echo "📋 샘플 데이터 파일을 사용합니다."
        cp students_sample.csv care_student.csv
    fi
fi

echo "🚀 애플리케이션을 시작합니다..."
echo "💻 브라우저에서 http://localhost:8501 로 접속하세요."
echo "🛑 종료하려면 Ctrl+C를 누르세요."
echo

$PYTHON_CMD -m streamlit run app.py