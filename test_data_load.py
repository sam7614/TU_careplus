#!/usr/bin/env python3
"""
데이터 로드 테스트 스크립트
"""

import pandas as pd
import os
from config import DATA_CONFIG

def test_data_loading():
    """데이터 로딩 테스트"""
    print("🎓 TIUM CARE+ 데이터 로딩 테스트")
    print("=" * 40)
    
    # Primary file test
    primary_file = DATA_CONFIG['primary_file']
    print(f"\n📋 Primary 파일 테스트: {primary_file}")
    
    if os.path.exists(primary_file):
        try:
            df = pd.read_csv(primary_file, encoding='utf-8')
            print(f"✅ 파일 로드 성공: {len(df)}행, {len(df.columns)}열")
            print(f"📊 컬럼: {list(df.columns)}")
            
            # Check required columns
            required_columns = DATA_CONFIG['required_columns']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"❌ 누락된 컬럼: {missing_columns}")
            else:
                print("✅ 모든 필수 컬럼 존재")
                
            # Show sample data
            print(f"\n📋 샘플 데이터 (첫 3행):")
            print(df.head(3).to_string())
            
        except Exception as e:
            print(f"❌ 파일 로드 오류: {str(e)}")
    else:
        print(f"❌ 파일 없음: {primary_file}")
    
    # Backup file test
    backup_file = DATA_CONFIG['backup_file']
    print(f"\n📋 Backup 파일 테스트: {backup_file}")
    
    if os.path.exists(backup_file):
        try:
            df = pd.read_csv(backup_file, encoding='utf-8')
            print(f"✅ 파일 로드 성공: {len(df)}행, {len(df.columns)}열")
            print(f"📊 컬럼: {list(df.columns)}")
        except Exception as e:
            print(f"❌ 파일 로드 오류: {str(e)}")
    else:
        print(f"❌ 파일 없음: {backup_file}")
    
    print("\n🎯 테스트 완료!")

if __name__ == "__main__":
    test_data_loading()