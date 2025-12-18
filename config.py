"""
Configuration file for Crisis Student Management System with Survival Analysis
생존분석 기반 위기 학생 관리 시스템 설정 파일
"""

# 위기 학생 판정 기준 (Crisis Student Criteria)
CRISIS_CRITERIA = {
    # 출석률 기준 (Attendance Rate Threshold)
    'attendance_threshold': 70.0,  # 70% 미만
    
    # 학점 기준 (GPA Threshold) 
    'gpa_threshold': 2.0,  # 2.0 미만
    
    # 등록금 미납 상태 (Unpaid Tuition Status)
    'unpaid_tuition_status': ['미납', '부분납']  # 미납 또는 부분납
}

# 생존분석 기준 (Survival Analysis Criteria)
SURVIVAL_CRITERIA = {
    # 위험 점수 임계값 (Risk Score Thresholds)
    'high_risk_threshold': 0.6,     # 60% 이상 위험
    'medium_risk_threshold': 0.35,  # 35-60% 위험
    'low_risk_threshold': 0.15,     # 15-35% 위험
    
    # 가중치 (Feature Weights)
    'weights': {
        'gpa': 0.20,           # 학점 가중치
        'attendance': 0.20,    # 출석률 가중치
        'tuition': 0.15,       # 등록금 가중치
        'counseling': 0.12,    # 상담 횟수 가중치
        'scholarship': 0.08,   # 장학금 가중치
        'library': 0.05,       # 도서관 이용 가중치
        'double_major': 0.08,  # 다전공신청 가중치
        'module': 0.07,        # 모듈신청 가중치
        'extracurricular': 0.05 # 비교과참여 가중치
    },
    
    # 시간 단위 (Time Units)
    'time_unit': 'semester',  # 학기 단위
    'max_time': 8            # 최대 8학기
}

# UI 설정 (UI Configuration)
UI_CONFIG = {
    # 페이지 제목 (Page Title)
    'page_title': 'TIUM CARE+',
    
    # 메트릭 라벨 (Metric Labels)
    'metrics': {
        'total_students': '📊 전체 학생 수',
        'high_risk_students': '🚨 고위험 학생 수',
        'medium_risk_students': '⚠️ 중위험 학생 수',
        'low_risk_students': '📈 저위험 학생 수',
        'average_risk_score': '📊 평균 위험 점수'
    },
    
    # 섹션 제목 (Section Titles)
    'sections': {
        'survival_analysis': '📈 생존분석 대시보드',
        'risk_distribution': '📊 위험도 분포',
        'high_risk_list': '🚨 고위험 학생 명단',
        'medium_risk_list': '⚠️ 중위험 학생 명단',
        'department_filter': '🏫 학과별 필터링',
        'survival_curves': '📈 생존 곡선',
        'risk_factors': '🔍 위험 요인 분석',
        'all_students': '📋 전체 학생 데이터 보기'
    }
}

# 데이터 파일 설정 (Data File Configuration)
DATA_CONFIG = {
    # 기본 데이터 파일 (Primary Data File)
    'primary_file': 'care_student.csv',
    
    # 백업 데이터 파일 (Backup Data File)
    'backup_file': 'students_sample.csv',
    
    # 지원하는 인코딩 (Supported Encodings)
    'encodings': ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig'],
    
    # 필수 컬럼 (Required Columns)
    'required_columns': [
        '학번', '이름', '학과', '학년', '직전학기_평점', '평균_출석률',
        '현재_성적', '상담_받은_횟수', '장학금_신청', '현재_평점',
        '도서관_이용_횟수', '등록금_납부_상태', '다전공신청', '모듈신청', '비교과참여횟수'
    ],
    
    # 숫자 컬럼 (Numeric Columns)
    'numeric_columns': [
        '직전학기_평점', '평균_출석률', '현재_성적', 
        '상담_받은_횟수', '현재_평점', '도서관_이용_횟수', '비교과참여횟수'
    ]
}

# 색상 테마 (Color Theme)
COLOR_THEME = {
    'primary': '#FF4B4B',      # Streamlit Red
    'secondary': '#0068C9',    # Streamlit Blue  
    'success': '#00D4AA',      # Streamlit Green
    'warning': '#FFBD45',      # Streamlit Orange
    'danger': '#FF4B4B',       # Red for crisis
    'info': '#0068C9'          # Blue for info
}

# 메시지 템플릿 (Message Templates)
MESSAGES = {
    'success': {
        'no_crisis_students': '✅ 현재 위기 학생이 없습니다!',
        'data_loaded': '✅ 데이터가 성공적으로 로드되었습니다.'
    },
    'error': {
        'file_not_found': '❌ 데이터 파일을 찾을 수 없습니다: {}',
        'encoding_error': '❌ 파일 인코딩을 읽을 수 없습니다. UTF-8 또는 CP949 인코딩으로 저장된 파일을 사용해주세요.',
        'column_mismatch': '❌ CSV 파일의 컬럼 수가 부족합니다. 필요: {}, 실제: {}',
        'general_error': '❌ 파일을 읽는 중 오류가 발생했습니다: {}'
    },
    'info': {
        'file_help': '💡 같은 폴더에 \'care_student.csv\' 파일이 있는지 확인해주세요.',
        'department_filter_help': '특정 학과의 학생들만 보려면 선택하세요.',
        'no_students_in_dept': '📋 \'{}\' 학과에는 위기 학생이 없습니다.',
        'using_sample_data': '📋 샘플 데이터를 사용합니다.'
    },
    'warning': {
        'crisis_students_found': '⚠️ 총 {}명의 위기 학생이 발견되었습니다.'
    }
}

# 위기 요인 메시지 (Risk Factor Messages)
RISK_FACTORS = {
    'attendance': '출석률 부족 ({:.1f}%)',
    'gpa': '학점 부족 ({:.1f})', 
    'tuition': '등록금 미납',
    'partial_tuition': '등록금 부분납',
    'counseling': '상담 부족 ({:.0f}회)',
    'scholarship': '장학금 미신청',
    'library': '도서관 이용 부족 ({:.0f}회)'
}

# 위험도 레벨 (Risk Levels)
RISK_LEVELS = {
    'high': {
        'label': '🚨 고위험',
        'color': '#FF4444',
        'description': '즉시 개입 필요'
    },
    'medium': {
        'label': '⚠️ 중위험', 
        'color': '#FF8800',
        'description': '주의 깊은 관찰 필요'
    },
    'low': {
        'label': '📈 저위험',
        'color': '#FFAA00', 
        'description': '예방적 지원 권장'
    },
    'safe': {
        'label': '✅ 안전',
        'color': '#00AA44',
        'description': '정상 범위'
    }
}

# 개발자 정보 (Developer Information)
DEVELOPER_INFO = {
    'name': 'Kiro AI Assistant',
    'version': '1.0.0',
    'last_updated': '2024년 12월',
    'description': '동명대학교 위기 학생 관리 시스템'
}