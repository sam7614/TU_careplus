#!/usr/bin/env python3
"""
Test script for the Crisis Student Management System
"""

import pandas as pd
import sys
import os

def test_data_loading():
    """Test data loading functionality"""
    print("🧪 Testing data loading...")
    
    # Test with sample file
    if os.path.exists('students_sample.csv'):
        try:
            df = pd.read_csv('students_sample.csv')
            print(f"✅ Sample data loaded successfully: {len(df)} rows")
            print(f"📊 Columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"❌ Error loading sample data: {e}")
            return None
    else:
        print("❌ Sample data file not found")
        return None

def test_crisis_identification(df):
    """Test crisis student identification"""
    if df is None:
        return
    
    print("\n🧪 Testing crisis student identification...")
    
    # Crisis criteria
    crisis_conditions = (
        (df['평균_출석률'] < 70) |
        (df['직전학기_평점'] < 2.0) |
        (df['등록금_납부_상태'] == '미납')
    )
    
    crisis_students = df[crisis_conditions]
    print(f"⚠️ Found {len(crisis_students)} crisis students out of {len(df)} total")
    
    if len(crisis_students) > 0:
        print("📋 Crisis students:")
        for _, student in crisis_students.iterrows():
            factors = []
            if student['평균_출석률'] < 70:
                factors.append(f"출석률 {student['평균_출석률']:.1f}%")
            if student['직전학기_평점'] < 2.0:
                factors.append(f"평점 {student['직전학기_평점']:.1f}")
            if student['등록금_납부_상태'] == '미납':
                factors.append("등록금 미납")
            
            print(f"  - {student['이름']} ({student['학과']}): {' | '.join(factors)}")

def test_metrics_calculation(df):
    """Test metrics calculation"""
    if df is None:
        return
    
    print("\n🧪 Testing metrics calculation...")
    
    total_students = len(df)
    crisis_students = len(df[(df['평균_출석률'] < 70) | (df['직전학기_평점'] < 2.0) | (df['등록금_납부_상태'] == '미납')])
    avg_attendance = df['평균_출석률'].mean()
    
    print(f"📊 Total students: {total_students}")
    print(f"⚠️ Crisis students: {crisis_students} ({crisis_students/total_students*100:.1f}%)")
    print(f"📈 Average attendance: {avg_attendance:.1f}%")

def main():
    """Main test function"""
    print("🎓 Crisis Student Management System - Test Suite")
    print("=" * 50)
    
    # Test data loading
    df = test_data_loading()
    
    # Test crisis identification
    test_crisis_identification(df)
    
    # Test metrics calculation
    test_metrics_calculation(df)
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    
    if df is not None:
        print("\n💡 To run the full application:")
        print("   streamlit run app.py")
        print("\n📝 Make sure you have the required packages installed:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()