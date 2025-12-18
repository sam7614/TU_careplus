#!/usr/bin/env python3
"""
Simple test to check if Python and pandas work
"""

print("🎓 동명대학교 위기 학생 관리 시스템 - 간단 테스트")
print("=" * 50)

try:
    import pandas as pd
    print("✅ Pandas 라이브러리 로드 성공")
    
    # Test with sample data
    import os
    if os.path.exists('students_sample.csv'):
        df = pd.read_csv('students_sample.csv')
        print(f"✅ 샘플 데이터 로드 성공: {len(df)}명의 학생")
        
        # Show basic info
        print(f"📊 컬럼: {list(df.columns)}")
        
        # Test crisis identification
        crisis_students = df[
            (df['평균_출석률'] < 70) |
            (df['직전학기_평점'] < 2.0) |
            (df['등록금_납부_상태'] == '미납')
        ]
        
        print(f"⚠️ 위기 학생: {len(crisis_students)}명 ({len(crisis_students)/len(df)*100:.1f}%)")
        print(f"📈 평균 출석률: {df['평균_출석률'].mean():.1f}%")
        
        if len(crisis_students) > 0:
            print("\n🚨 위기 학생 목록:")
            for _, student in crisis_students.head(5).iterrows():
                factors = []
                if student['평균_출석률'] < 70:
                    factors.append(f"출석률 {student['평균_출석률']:.1f}%")
                if student['직전학기_평점'] < 2.0:
                    factors.append(f"평점 {student['직전학기_평점']:.1f}")
                if student['등록금_납부_상태'] == '미납':
                    factors.append("등록금 미납")
                
                print(f"  - {student['이름']} ({student['학과']}): {' | '.join(factors)}")
        
        print("\n✅ 모든 기능이 정상 작동합니다!")
        print("💡 Streamlit 설치 후 'streamlit run app.py'로 웹 버전을 실행하세요.")
        
    else:
        print("❌ 샘플 데이터 파일을 찾을 수 없습니다.")
        
except ImportError as e:
    print(f"❌ 라이브러리 로드 실패: {e}")
    print("💡 'pip install pandas' 명령어로 pandas를 설치하세요.")
except Exception as e:
    print(f"❌ 오류 발생: {e}")

print("=" * 50)