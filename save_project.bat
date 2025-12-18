@echo off
echo 🎓 동명대학교 생존분석 기반 위기 학생 관리 시스템 - 프로젝트 저장
echo ================================================================

REM 현재 날짜와 시간으로 백업 폴더명 생성
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%" & set "Sec=%dt:~12,2%"
set "datestamp=%YYYY%%MM%%DD%_%HH%%Min%%Sec%"

set "backup_name=crisis_student_system_%datestamp%"

echo 📦 백업 폴더 생성: %backup_name%
mkdir "%backup_name%"

echo 📋 필수 파일들 복사 중...
copy "app.py" "%backup_name%\"
copy "config.py" "%backup_name%\"
copy "requirements.txt" "%backup_name%\"
copy "students_sample.csv" "%backup_name%\"
copy "care_student.csv" "%backup_name%\"
copy "README.md" "%backup_name%\"
copy "PROJECT_SUMMARY.md" "%backup_name%\"
copy "DEPLOYMENT.md" "%backup_name%\"
copy "test_app.py" "%backup_name%\"
copy "simple_test.py" "%backup_name%\"
copy "run_app.bat" "%backup_name%\"
copy "run_app.sh" "%backup_name%\"

echo 📁 .kiro 폴더 복사 중...
xcopy ".kiro" "%backup_name%\.kiro" /E /I /H

echo 🗜️ ZIP 파일로 압축 중...
powershell -command "Compress-Archive -Path '%backup_name%' -DestinationPath '%backup_name%.zip' -Force"

echo ✅ 프로젝트 저장 완료!
echo 📍 저장 위치: %backup_name%.zip
echo 📍 폴더 위치: %backup_name%\

echo.
echo 💡 이 파일을 다른 컴퓨터로 옮기거나 백업용으로 사용하세요.
echo 💡 복원할 때는 ZIP 파일을 압축 해제하고 restore_project.bat를 실행하세요.

pause