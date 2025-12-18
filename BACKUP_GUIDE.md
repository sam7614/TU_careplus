# 🎓 동명대학교 생존분석 기반 위기 학생 관리 시스템
## 프로젝트 백업 및 복원 가이드

---

## 📦 **프로젝트 저장 방법**

### 방법 1: 자동 백업 스크립트 사용 (권장)

1. **`save_project.bat`** 파일을 더블클릭하여 실행
2. 자동으로 날짜/시간이 포함된 백업 폴더와 ZIP 파일이 생성됩니다
3. 생성된 ZIP 파일을 안전한 곳에 보관하세요

**생성되는 파일:**
- `crisis_student_system_YYYYMMDD_HHMMSS.zip` (압축 파일)
- `crisis_student_system_YYYYMMDD_HHMMSS/` (폴더)

### 방법 2: 수동 백업

다음 파일들을 복사하여 백업:

**필수 파일:**
- `app.py` - 메인 애플리케이션
- `config.py` - 설정 파일
- `requirements.txt` - 패키지 의존성
- `students_sample.csv` - 샘플 데이터
- `care_student.csv` - 메인 데이터
- `README.md` - 프로젝트 설명서

**선택적 파일:**
- `test_app.py` - 테스트 스크립트
- `simple_test.py` - 간단한 테스트
- `DEPLOYMENT.md` - 배포 가이드
- `PROJECT_SUMMARY.md` - 프로젝트 요약
- `.kiro/` 폴더 - 스펙 문서들

---

## 📥 **프로젝트 복원 방법**

### 방법 1: 자동 복원 스크립트 사용 (권장)

1. 백업한 ZIP 파일을 원하는 위치에 압축 해제
2. 압축 해제된 폴더로 이동
3. **`restore_project.bat`** 파일을 더블클릭하여 실행
4. 스크립트가 자동으로:
   - Python 설치 확인
   - 필요한 패키지 설치
   - 데이터 파일 확인
   - 프로젝트 준비 완료

### 방법 2: 수동 복원

1. **Python 설치 확인**
   ```cmd
   python --version
   ```

2. **패키지 설치**
   ```cmd
   pip install -r requirements.txt
   ```

3. **애플리케이션 실행**
   ```cmd
   streamlit run app.py
   ```

---

## 🚀 **애플리케이션 실행**

복원 완료 후 다음 방법으로 실행:

### Windows:
```cmd
streamlit run app.py
```
또는
```cmd
run_app.bat
```

### 브라우저 접속:
- **로컬**: http://localhost:8501
- **네트워크**: http://[IP주소]:8501

---

## 📋 **시스템 요구사항**

### 필수 요구사항:
- **Python 3.8 이상**
- **pip (Python 패키지 관리자)**
- **인터넷 연결** (패키지 설치용)

### 필요한 Python 패키지:
- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- plotly
- lifelines
- openpyxl

---

## 🔧 **문제 해결**

### Python이 설치되지 않은 경우:
1. https://www.python.org/downloads/ 방문
2. Python 3.8 이상 버전 다운로드
3. 설치 시 "Add Python to PATH" 옵션 체크

### 패키지 설치 오류:
```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 포트 충돌 오류:
```cmd
streamlit run app.py --server.port 8502
```

### 데이터 파일 오류:
- `students_sample.csv` 파일이 있는지 확인
- 파일 인코딩이 UTF-8인지 확인

---

## 📞 **지원 및 문의**

프로젝트 관련 문의사항이 있으시면:
1. `README.md` 파일 참조
2. `DEPLOYMENT.md` 파일 참조
3. 로그 파일 확인

---

## 📝 **버전 정보**

- **프로젝트명**: 동명대학교 생존분석 기반 위기 학생 관리 시스템
- **버전**: 1.0.0
- **최종 업데이트**: 2024년 12월
- **개발자**: Kiro AI Assistant

---

**💡 팁**: 정기적으로 백업을 생성하여 데이터 손실을 방지하세요!