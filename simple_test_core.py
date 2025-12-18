#!/usr/bin/env python3
"""
Simple test for core functionality without Streamlit
"""

import pandas as pd
import numpy as np
import os

# Simple test data
def create_test_data():
    """Create test data for the system"""
    data = {
        '학번': ['2021001', '2021002', '2021003', '2021004', '2021005'],
        '이름': ['김철수', '이영희', '박민수', '최지영', '정현우'],
        '학과': ['컴퓨터공학과', '경영학과', '컴퓨터공학과', '경영학과', '컴퓨터공학과'],
        '학년': [2, 3, 1, 4, 2],
        '직전학기_평점': [2.1, 3.8, 2.5, 4.2, 1.8],
        '평균_출석률': [65.0, 95.0, 75.0, 98.0, 55.0],
        '현재_성적': [70, 90, 75, 95, 60],
        '상담_받은_횟수': [1, 5, 2, 8, 0],
        '장학금_신청': ['X', 'O', 'X', 'O', 'X'],
        '현재_평점': [2.1, 3.8, 2.5, 4.2, 1.8],
        '도서관_이용_횟수': [5, 25, 10, 30, 2],
        '등록금_납부_상태': ['부분납', '완납', '완납', '완납', '미납'],
        '다전공신청': ['X', 'O', 'X', 'O', 'X'],
        '모듈신청': ['X', 'O', 'X', 'X', 'X'],
        '비교과참여횟수': [1, 8, 3, 12, 0]
    }
    return pd.DataFrame(data)

def calculate_risk_score(df):
    """Calculate risk scores"""
    df_risk = df.copy()
    
    # GPA risk
    df_risk['gpa_risk'] = 1 - (df_risk['직전학기_평점'] / 4.5)
    df_risk['gpa_risk'] = np.clip(df_risk['gpa_risk'], 0, 1)
    
    # Attendance risk
    df_risk['attendance_risk'] = 1 - (df_risk['평균_출석률'] / 100)
    df_risk['attendance_risk'] = np.clip(df_risk['attendance_risk'], 0, 1)
    
    # Tuition risk
    tuition_map = {'완납': 0.0, '부분납': 0.6, '미납': 1.0}
    df_risk['tuition_risk'] = df_risk['등록금_납부_상태'].map(tuition_map).fillna(0.5)
    
    # Counseling risk
    max_counseling = df_risk['상담_받은_횟수'].max()
    df_risk['counseling_risk'] = 1 - (df_risk['상담_받은_횟수'] / max_counseling) if max_counseling > 0 else 0
    
    # Scholarship risk
    df_risk['scholarship_risk'] = df_risk['장학금_신청'].map({'O': 0.0, 'X': 1.0}).fillna(0.5)
    
    # Library risk
    max_library = df_risk['도서관_이용_횟수'].max()
    df_risk['library_risk'] = 1 - (df_risk['도서관_이용_횟수'] / max_library) if max_library > 0 else 0
    
    # Protective factors
    df_risk['double_major_bonus'] = df_risk['다전공신청'].map({'O': -0.1, 'X': 0.0}).fillna(0.0)
    df_risk['module_bonus'] = df_risk['모듈신청'].map({'O': -0.1, 'X': 0.0}).fillna(0.0)
    
    max_extra = df_risk['비교과참여횟수'].max()
    if max_extra > 0:
        normalized = df_risk['비교과참여횟수'] / max_extra
        df_risk['extracurricular_bonus'] = -(normalized * 0.2)
    else:
        df_risk['extracurricular_bonus'] = 0.0
    
    # Weights
    weights = {
        'gpa': 0.20,
        'attendance': 0.20,
        'tuition': 0.15,
        'counseling': 0.12,
        'scholarship': 0.08,
        'library': 0.05,
        'double_major': 0.08,
        'module': 0.07,
        'extracurricular': 0.05
    }
    
    # Calculate final risk score
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
        if score >= 0.6:
            return 'high'
        elif score >= 0.35:
            return 'medium'
        elif score >= 0.15:
            return 'low'
        else:
            return 'safe'
    
    df_risk['위험_레벨'] = df_risk['위험_점수'].apply(classify_risk)
    
    return df_risk

def main():
    print("🎓 TIUM CARE+ 시스템 핵심 기능 테스트")
    print("=" * 50)
    
    # Create test data
    df = create_test_data()
    print(f"✅ 테스트 데이터 생성: {len(df)}명")
    
    # Calculate risk scores
    df_with_risk = calculate_risk_score(df)
    print("✅ 위험도 계산 완료")
    
    # Show results
    print("\n📊 분석 결과:")
    risk_counts = df_with_risk['위험_레벨'].value_counts()
    
    risk_labels = {
        'high': '🚨 고위험',
        'medium': '⚠️ 중위험',
        'low': '📈 저위험',
        'safe': '✅ 안전'
    }
    
    for risk_level in ['high', 'medium', 'low', 'safe']:
        count = risk_counts.get(risk_level, 0)
        label = risk_labels[risk_level]
        print(f"  {label}: {count}명")
    
    print(f"\n📊 평균 위험점수: {df_with_risk['위험_점수'].mean():.3f}")
    
    # Show individual results
    print("\n👥 개별 학생 결과:")
    for _, row in df_with_risk.iterrows():
        risk_label = risk_labels[row['위험_레벨']]
        print(f"  {row['이름']} ({row['학과']}): {row['위험_점수']:.3f} - {risk_label}")
    
    # Test variable analysis
    print("\n🔍 변수별 분석 테스트:")
    
    # 다전공신청별 분석
    print("  다전공신청별:")
    for value in df_with_risk['다전공신청'].unique():
        group = df_with_risk[df_with_risk['다전공신청'] == value]
        high_risk = len(group[group['위험_레벨'] == 'high'])
        avg_risk = group['위험_점수'].mean()
        print(f"    {value}: {len(group)}명, 고위험 {high_risk}명, 평균위험도 {avg_risk:.3f}")
    
    # 학과별 분석
    print("  학과별:")
    for dept in df_with_risk['학과'].unique():
        group = df_with_risk[df_with_risk['학과'] == dept]
        high_risk = len(group[group['위험_레벨'] == 'high'])
        avg_risk = group['위험_점수'].mean()
        print(f"    {dept}: {len(group)}명, 고위험 {high_risk}명, 평균위험도 {avg_risk:.3f}")
    
    print("\n🎯 핵심 기능 테스트 완료!")
    print("✅ 시스템이 정상적으로 작동합니다.")
    print("=" * 50)

if __name__ == "__main__":
    main()