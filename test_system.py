#!/usr/bin/env python3
"""
Test script for Crisis Student Management System
위기 학생 관리 시스템 테스트 스크립트
"""

import pandas as pd
import numpy as np
import os
from config import DATA_CONFIG, SURVIVAL_CRITERIA, RISK_LEVELS

def load_student_data(file_path: str) -> pd.DataFrame:
    """Load student data from CSV file"""
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return pd.DataFrame()
    
    try:
        # Try different encodings for Korean text
        df = None
        for encoding in DATA_CONFIG['encodings']:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"✅ 파일 로드 성공 (인코딩: {encoding})")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            print("❌ 인코딩 오류")
            return pd.DataFrame()
        
        # Set column names
        expected_columns = DATA_CONFIG['required_columns']
        if len(df.columns) >= len(expected_columns):
            df.columns = expected_columns
        else:
            print(f"❌ 컬럼 수 부족: 필요 {len(expected_columns)}, 실제 {len(df.columns)}")
            return pd.DataFrame()
        
        # Convert numeric columns
        for col in DATA_CONFIG['numeric_columns']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        print(f"❌ 파일 로드 오류: {str(e)}")
        return pd.DataFrame()

def calculate_survival_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate survival-based risk scores"""
    if df.empty:
        return df
    
    df_risk = df.copy()
    
    # Calculate risk components
    df_risk['gpa_risk'] = 1 - (df_risk['직전학기_평점'] / 4.5)
    df_risk['gpa_risk'] = np.clip(df_risk['gpa_risk'], 0, 1)
    
    df_risk['attendance_risk'] = 1 - (df_risk['평균_출석률'] / 100)
    df_risk['attendance_risk'] = np.clip(df_risk['attendance_risk'], 0, 1)
    
    tuition_risk_map = {'완납': 0.0, '부분납': 0.6, '미납': 1.0}
    df_risk['tuition_risk'] = df_risk['등록금_납부_상태'].map(tuition_risk_map).fillna(0.5)
    
    max_counseling = df_risk['상담_받은_횟수'].max() if df_risk['상담_받은_횟수'].max() > 0 else 1
    df_risk['counseling_risk'] = 1 - (df_risk['상담_받은_횟수'] / max_counseling)
    df_risk['counseling_risk'] = np.clip(df_risk['counseling_risk'], 0, 1)
    
    df_risk['scholarship_risk'] = df_risk['장학금_신청'].map({'O': 0.0, 'X': 1.0}).fillna(0.5)
    
    max_library = df_risk['도서관_이용_횟수'].max() if df_risk['도서관_이용_횟수'].max() > 0 else 1
    df_risk['library_risk'] = 1 - (df_risk['도서관_이용_횟수'] / max_library)
    df_risk['library_risk'] = np.clip(df_risk['library_risk'], 0, 1)
    
    # Protective factors (bonuses)
    if '다전공신청' in df_risk.columns:
        df_risk['double_major_bonus'] = df_risk['다전공신청'].map({'O': -0.1, 'X': 0.0}).fillna(0.0)
    else:
        df_risk['double_major_bonus'] = 0.0
    
    if '모듈신청' in df_risk.columns:
        df_risk['module_bonus'] = df_risk['모듈신청'].map({'O': -0.1, 'X': 0.0}).fillna(0.0)
    else:
        df_risk['module_bonus'] = 0.0
    
    if '비교과참여횟수' in df_risk.columns:
        max_extracurricular = df_risk['비교과참여횟수'].max() if df_risk['비교과참여횟수'].max() > 0 else 1
        normalized_participation = df_risk['비교과참여횟수'] / max_extracurricular
        df_risk['extracurricular_bonus'] = -(normalized_participation * 0.2)
        df_risk['extracurricular_bonus'] = np.clip(df_risk['extracurricular_bonus'], -0.2, 0)
    else:
        df_risk['extracurricular_bonus'] = 0.0
    
    # Calculate weighted risk score
    weights = SURVIVAL_CRITERIA['weights']
    df_risk['위험_점수'] = (
        df_risk['gpa_risk'] * weights['gpa'] +
        df_risk['attendance_risk'] * weights['attendance'] +
        df_risk['tuition_risk'] * weights['tuition'] +
        df_risk['counseling_risk'] * weights['counseling'] +
        df_risk['scholarship_risk'] * weights['scholarship'] +
        df_risk['library_risk'] * weights['library'] +
        df_risk['double_major_bonus'] * weights['double_major'] +
        df_risk['module_bonus'] * weights['module'] +
        df_risk['extracurricular_bonus'] * weights['extracurricular']
    )
    
    # Classify risk levels
    def classify_risk(score):
        if score >= SURVIVAL_CRITERIA['high_risk_threshold']:
            return 'high'
        elif score >= SURVIVAL_CRITERIA['medium_risk_threshold']:
            return 'medium'
        elif score >= SURVIVAL_CRITERIA['low_risk_threshold']:
            return 'low'
        else:
            return 'safe'
    
    df_risk['위험_레벨'] = df_risk['위험_점수'].apply(classify_risk)
    
    return df_risk

def test_system():
    """Test the crisis student management system"""
    print("🎓 TIUM CARE+ 위기 학생 관리 시스템 테스트")
    print("=" * 50)
    
    # Load data
    print("\n📊 데이터 로드 테스트:")
    df = load_student_data(DATA_CONFIG['primary_file'])
    if df.empty and os.path.exists(DATA_CONFIG['backup_file']):
        print("📋 백업 데이터 사용")
        df = load_student_data(DATA_CONFIG['backup_file'])
    
    if df.empty:
        print("❌ 데이터를 로드할 수 없습니다.")
        return
    
    print(f"✅ 데이터 로드 완료: {len(df)}명의 학생 데이터")
    print(f"📋 컬럼: {list(df.columns)}")
    
    # Calculate risk scores
    print("\n🔍 위험도 계산 테스트:")
    df_with_risk = calculate_survival_risk_score(df)
    
    if '위험_점수' not in df_with_risk.columns:
        print("❌ 위험도 계산 실패")
        return
    
    print("✅ 위험도 계산 완료")
    
    # Analyze results
    print("\n📈 분석 결과:")
    risk_counts = df_with_risk['위험_레벨'].value_counts()
    total_students = len(df_with_risk)
    
    print(f"📊 전체 학생 수: {total_students}명")
    
    for risk_level in ['high', 'medium', 'low', 'safe']:
        count = risk_counts.get(risk_level, 0)
        percentage = (count / total_students * 100) if total_students > 0 else 0
        label = RISK_LEVELS[risk_level]['label']
        print(f"  {label}: {count}명 ({percentage:.1f}%)")
    
    avg_risk = df_with_risk['위험_점수'].mean()
    print(f"📊 평균 위험점수: {avg_risk:.3f}")
    
    # Test variable-specific analysis
    print("\n🔍 변수별 분석 테스트:")
    
    variables_to_test = ['다전공신청', '모듈신청', '비교과참여횟수', '학과', '학년']
    
    for variable in variables_to_test:
        if variable in df_with_risk.columns:
            unique_values = df_with_risk[variable].dropna().unique()
            unique_values = [val for val in unique_values if pd.notna(val)]
            print(f"  {variable}: {len(unique_values)}개 그룹 ({unique_values[:3]}{'...' if len(unique_values) > 3 else ''})")
            
            # Test one group analysis
            if len(unique_values) > 0:
                test_value = unique_values[0]
                group_df = df_with_risk[df_with_risk[variable] == test_value]
                group_risk_counts = group_df['위험_레벨'].value_counts()
                high_risk_count = group_risk_counts.get('high', 0)
                print(f"    {variable}={test_value}: {len(group_df)}명 (고위험 {high_risk_count}명)")
        else:
            print(f"  {variable}: ❌ 데이터에 없음")
    
    # Test high-risk students
    print("\n🚨 고위험 학생 분석:")
    high_risk_students = df_with_risk[df_with_risk['위험_레벨'] == 'high']
    
    if len(high_risk_students) > 0:
        print(f"✅ 고위험 학생 {len(high_risk_students)}명 발견")
        print("상위 3명:")
        top_3 = high_risk_students.nlargest(3, '위험_점수')[['이름', '학과', '학년', '위험_점수']]
        for idx, row in top_3.iterrows():
            print(f"  - {row['이름']} ({row['학과']} {row['학년']}학년): {row['위험_점수']:.3f}")
    else:
        print("✅ 고위험 학생 없음 (시스템이 잘 조정됨)")
    
    print("\n🎯 시스템 테스트 완료!")
    print("=" * 50)

if __name__ == "__main__":
    test_system()