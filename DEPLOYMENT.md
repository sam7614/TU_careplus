# 🚀 배포 가이드

## 📋 시스템 요구사항

- **Python 3.7 이상**
- **Windows 10/11** 또는 **macOS/Linux**
- **최소 2GB RAM**
- **인터넷 연결** (패키지 설치용)

## 🛠️ 설치 방법

### Windows 사용자

1. **Python 설치**
   - [Python 공식 사이트](https://www.python.org/downloads/)에서 최신 버전 다운로드
   - 설치 시 "Add Python to PATH" 옵션 체크 필수

2. **프로젝트 실행**
   ```cmd
   # 배치 파일 실행 (권장)
   run_app.bat
   
   # 또는 수동 실행
   pip install -r requirements.txt
   streamlit run app.py
   ```

### macOS/Linux 사용자

1. **Python 설치 확인**
   ```bash
   python3 --version
   # 또는
   python --version
   ```

2. **프로젝트 실행**
   ```bash
   # 셸 스크립트 실행 (권장)
   ./run_app.sh
   
   # 또는 수동 실행
   pip3 install -r requirements.txt
   python3 -m streamlit run app.py
   ```

## 📊 데이터 파일 준비

### 1. 파일명
- `care_student.csv` (기본)
- `students_sample.csv` (테스트용)

### 2. 인코딩
- **UTF-8** (권장)
- **CP949** (한국어 Windows 환경)

### 3. Excel에서 CSV 저장하기
1. Excel에서 데이터 준비
2. **파일 → 다른 이름으로 저장**
3. **파일 형식: CSV UTF-8 (쉼표로 분리)(*.csv)** 선택
4. 파일명을 `care_student.csv`로 저장

## 🌐 웹 배포

### Streamlit Cloud 배포

1. **GitHub 저장소 생성**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Streamlit Cloud 연결**
   - [share.streamlit.io](https://share.streamlit.io) 접속
   - GitHub 계정으로 로그인
   - 저장소 선택 및 배포

### Heroku 배포

1. **추가 파일 생성**
   ```bash
   # Procfile
   echo "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile
   
   # runtime.txt
   echo "python-3.9.16" > runtime.txt
   ```

2. **Heroku CLI로 배포**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

## 🔧 문제 해결

### 일반적인 오류

1. **ModuleNotFoundError**
   ```bash
   pip install -r requirements.txt
   ```

2. **UnicodeDecodeError**
   - CSV 파일을 UTF-8로 다시 저장
   - 메모장에서 열어서 "UTF-8로 저장"

3. **FileNotFoundError**
   - `care_student.csv` 파일이 `app.py`와 같은 폴더에 있는지 확인
   - 파일명 대소문자 확인

4. **Port already in use**
   ```bash
   streamlit run app.py --server.port 8502
   ```

### Windows 특정 문제

1. **Python 명령어 인식 안됨**
   - Python 설치 시 PATH 추가 옵션 체크
   - 환경변수에 Python 경로 수동 추가

2. **권한 오류**
   - 관리자 권한으로 명령 프롬프트 실행
   - 또는 사용자 디렉토리에서 실행

### macOS/Linux 특정 문제

1. **Permission denied**
   ```bash
   chmod +x run_app.sh
   ./run_app.sh
   ```

2. **Python3 vs Python**
   - `python3` 명령어 사용
   - 또는 `alias python=python3` 설정

## 📈 성능 최적화

### 대용량 데이터 처리

1. **청크 단위 처리**
   ```python
   # app.py에서 대용량 파일 처리
   chunk_size = 1000
   for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
       process_chunk(chunk)
   ```

2. **캐싱 활용**
   ```python
   @st.cache_data
   def load_data():
       return pd.read_csv('care_student.csv')
   ```

### 메모리 사용량 최적화

1. **불필요한 컬럼 제거**
2. **데이터 타입 최적화**
3. **필터링 후 처리**

## 🔒 보안 고려사항

### 데이터 보호

1. **민감한 정보 마스킹**
   ```python
   # 학번 마스킹 예시
   df['학번_마스킹'] = df['학번'].str[:4] + '****'
   ```

2. **접근 제한**
   - Streamlit Cloud에서 비공개 저장소 사용
   - 인증 시스템 추가 고려

3. **데이터 암호화**
   - 중요한 데이터는 암호화하여 저장
   - HTTPS 사용 권장

## 📞 지원

### 기술 지원
- **이슈 리포팅**: GitHub Issues 활용
- **문서 개선**: Pull Request 환영

### 연락처
- **개발자**: Kiro AI Assistant
- **버전**: 1.0.0
- **최종 업데이트**: 2024년 12월

---

**참고**: 이 시스템은 교육 목적으로 개발되었으며, 실제 운영 환경에서는 추가적인 보안 및 성능 검토가 필요합니다.