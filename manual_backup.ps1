# TIUM CARE+ 프로젝트 수동 백업 스크립트
$datestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backup_name = "crisis_student_system_$datestamp"

Write-Host "🎓 TIUM CARE+ 프로젝트 백업 시작" -ForegroundColor Green
Write-Host "📦 백업 폴더: $backup_name" -ForegroundColor Yellow

# 백업 폴더 생성
New-Item -ItemType Directory -Name $backup_name -Force | Out-Null

# 필수 파일들 복사
$files = @(
    "app.py",
    "config.py", 
    "requirements.txt",
    "students_sample.csv",
    "care_student.csv",
    "README.md",
    "PROJECT_SUMMARY.md",
    "DEPLOYMENT.md",
    "test_app.py",
    "simple_test.py",
    "run_app.bat",
    "run_app.sh",
    "BACKUP_GUIDE.md",
    "save_project.bat",
    "restore_project.bat"
)

Write-Host "📋 파일 복사 중..." -ForegroundColor Cyan
foreach ($file in $files) {
    if (Test-Path $file) {
        Copy-Item $file "$backup_name\" -Force
        Write-Host "  ✅ $file" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️ $file (파일 없음)" -ForegroundColor Yellow
    }
}

# .kiro 폴더 복사
if (Test-Path ".kiro") {
    Write-Host "📁 .kiro 폴더 복사 중..." -ForegroundColor Cyan
    Copy-Item ".kiro" "$backup_name\.kiro" -Recurse -Force
    Write-Host "  ✅ .kiro 폴더 복사 완료" -ForegroundColor Green
}

# .streamlit 폴더 복사
if (Test-Path ".streamlit") {
    Write-Host "📁 .streamlit 폴더 복사 중..." -ForegroundColor Cyan
    Copy-Item ".streamlit" "$backup_name\.streamlit" -Recurse -Force
    Write-Host "  ✅ .streamlit 폴더 복사 완료" -ForegroundColor Green
}

# ZIP 파일로 압축
Write-Host "🗜️ ZIP 파일로 압축 중..." -ForegroundColor Cyan
Compress-Archive -Path $backup_name -DestinationPath "$backup_name.zip" -Force

Write-Host ""
Write-Host "✅ 프로젝트 백업 완료!" -ForegroundColor Green
Write-Host "📍 ZIP 파일: $backup_name.zip" -ForegroundColor Yellow
Write-Host "📍 폴더: $backup_name\" -ForegroundColor Yellow
Write-Host ""
Write-Host "💡 이 파일들을 안전한 곳에 보관하세요." -ForegroundColor Cyan
Write-Host "💡 복원할 때는 ZIP 파일을 압축 해제하고 restore_project.bat를 실행하세요." -ForegroundColor Cyan

# 백업 정보 파일 생성
$backup_info = @"
# TIUM CARE+ 프로젝트 백업 정보

**백업 생성일**: $(Get-Date -Format "yyyy년 MM월 dd일 HH:mm:ss")
**백업 이름**: $backup_name
**시스템 버전**: TIUM CARE+ v1.0.0

## 포함된 파일들:
- app.py (메인 애플리케이션)
- config.py (설정 파일)
- requirements.txt (패키지 의존성)
- students_sample.csv (샘플 데이터)
- care_student.csv (실제 데이터)
- README.md (프로젝트 설명)
- PROJECT_SUMMARY.md (프로젝트 요약)
- DEPLOYMENT.md (배포 가이드)
- .kiro/ (Kiro 설정 및 스펙 파일들)
- .streamlit/ (Streamlit 설정)

## 복원 방법:
1. ZIP 파일을 원하는 위치에 압축 해제
2. 해당 폴더에서 restore_project.bat 실행
3. 또는 run_app.bat 실행하여 바로 시작

## 시스템 요구사항:
- Python 3.8 이상
- 필요 패키지: streamlit, pandas, plotly, lifelines, scikit-learn 등

## 주요 기능:
- 생존분석 기반 위험도 예측
- 변수별 위험 분석 (다전공신청, 모듈신청, 비교과참여 등)
- 실시간 대시보드
- 개별 학생 상세 레포트
- 보호요인 시스템
"@

$backup_info | Out-File "$backup_name\BACKUP_INFO.md" -Encoding UTF8

Write-Host "📄 백업 정보 파일 생성: BACKUP_INFO.md" -ForegroundColor Green