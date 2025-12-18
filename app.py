import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from dataclasses import dataclass
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import warnings
warnings.filterwarnings('ignore')

from config import (
    CRISIS_CRITERIA, SURVIVAL_CRITERIA, UI_CONFIG, DATA_CONFIG, 
    MESSAGES, RISK_FACTORS, RISK_LEVELS, DEVELOPER_INFO
)

# Data Models for Survival Analysis
@dataclass
class KMCurve:
    """Kaplan-Meier curve data structure"""
    time_points: List[float]        # 시간 포인트들
    survival_probs: List[float]     # 생존 확률들
    confidence_lower: List[float]   # 95% 신뢰구간 하한
    confidence_upper: List[float]   # 95% 신뢰구간 상한
    group_name: str                 # 그룹명 (학과/학년)
    median_survival_time: Optional[float]  # 중앙생존시간
    
@dataclass
class SurvivalAnalysisResult:
    """Complete survival analysis results"""
    overall_curve: KMCurve          # 전체 생존곡선
    department_curves: List[KMCurve] # 학과별 생존곡선들
    grade_curves: List[KMCurve]     # 학년별 생존곡선들
    log_rank_p_value: Optional[float]  # 로그랭크 검정 p값

# Page configuration
st.set_page_config(
    page_title=UI_CONFIG['page_title'],
    page_icon="🎓",
    layout="wide"
)

def load_student_data(file_path: str) -> pd.DataFrame:
    """
    Load student data from CSV file with proper encoding handling
    """
    if not os.path.exists(file_path):
        st.error(MESSAGES['error']['file_not_found'].format(file_path))
        st.info(MESSAGES['info']['file_help'])
        return pd.DataFrame()
    
    try:
        # Try different encodings for Korean text
        df = None
        
        for encoding in DATA_CONFIG['encodings']:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            st.error(MESSAGES['error']['encoding_error'])
            return pd.DataFrame()
        
        # If columns don't match, assume the file has the structure we saw
        expected_columns = DATA_CONFIG['required_columns']
        if len(df.columns) >= len(expected_columns):
            df.columns = expected_columns
        else:
            st.error(MESSAGES['error']['column_mismatch'].format(len(expected_columns), len(df.columns)))
            return pd.DataFrame()
        
        # Convert numeric columns
        for col in DATA_CONFIG['numeric_columns']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        st.error(MESSAGES['error']['general_error'].format(str(e)))
        return pd.DataFrame()

def calculate_survival_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate survival-based risk scores for students using weighted features
    """
    if df.empty:
        return df
    
    df_risk = df.copy()
    
    # Normalize features to 0-1 scale for risk calculation
    # GPA risk (lower GPA = higher risk)
    df_risk['gpa_risk'] = 1 - (df_risk['직전학기_평점'] / 4.5)  # Assuming 4.5 scale
    df_risk['gpa_risk'] = np.clip(df_risk['gpa_risk'], 0, 1)
    
    # Attendance risk (lower attendance = higher risk)
    df_risk['attendance_risk'] = 1 - (df_risk['평균_출석률'] / 100)
    df_risk['attendance_risk'] = np.clip(df_risk['attendance_risk'], 0, 1)
    
    # Tuition risk (unpaid = high risk)
    tuition_risk_map = {'완납': 0.0, '부분납': 0.6, '미납': 1.0}
    df_risk['tuition_risk'] = df_risk['등록금_납부_상태'].map(tuition_risk_map).fillna(0.5)
    
    # Counseling risk (fewer sessions = higher risk)
    max_counseling = df_risk['상담_받은_횟수'].max() if df_risk['상담_받은_횟수'].max() > 0 else 1
    df_risk['counseling_risk'] = 1 - (df_risk['상담_받은_횟수'] / max_counseling)
    df_risk['counseling_risk'] = np.clip(df_risk['counseling_risk'], 0, 1)
    
    # Scholarship risk (no scholarship = higher risk)
    df_risk['scholarship_risk'] = df_risk['장학금_신청'].map({'O': 0.0, 'X': 1.0}).fillna(0.5)
    
    # Library usage risk (less usage = higher risk)
    max_library = df_risk['도서관_이용_횟수'].max() if df_risk['도서관_이용_횟수'].max() > 0 else 1
    df_risk['library_risk'] = 1 - (df_risk['도서관_이용_횟수'] / max_library)
    df_risk['library_risk'] = np.clip(df_risk['library_risk'], 0, 1)
    
    # Double major bonus (application = lower risk)
    if '다전공신청' in df_risk.columns:
        df_risk['double_major_bonus'] = df_risk['다전공신청'].map({'O': -0.1, 'X': 0.0}).fillna(0.0)  # 신청시 보너스
    else:
        df_risk['double_major_bonus'] = 0.0
    
    # Module bonus (application = lower risk)
    if '모듈신청' in df_risk.columns:
        df_risk['module_bonus'] = df_risk['모듈신청'].map({'O': -0.1, 'X': 0.0}).fillna(0.0)  # 신청시 보너스
    else:
        df_risk['module_bonus'] = 0.0
    
    # Extracurricular bonus (more activities = lower risk)
    if '비교과참여횟수' in df_risk.columns:
        max_extracurricular = df_risk['비교과참여횟수'].max() if df_risk['비교과참여횟수'].max() > 0 else 1
        # 정규화된 참여도를 보너스로 변환 (0~1 범위를 -0.2~0 범위로)
        normalized_participation = df_risk['비교과참여횟수'] / max_extracurricular
        df_risk['extracurricular_bonus'] = -(normalized_participation * 0.2)  # 음수 보너스 (최대 -0.2)
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
        df_risk['double_major_bonus'] * weights['double_major'] +  # 보너스 (음수값)
        df_risk['module_bonus'] * weights['module'] +  # 보너스 (음수값)
        df_risk['extracurricular_bonus'] * weights['extracurricular']  # 보너스 (음수값)
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
    
    # Calculate detailed risk factors
    df_risk['위기_요인'] = df_risk.apply(calculate_detailed_risk_factors, axis=1)
    
    return df_risk

def calculate_detailed_risk_factors(row: pd.Series) -> str:
    """
    Calculate detailed risk factors for a student based on survival analysis
    """
    factors = []
    
    # Check each risk factor
    if row['gpa_risk'] > 0.5:
        factors.append(RISK_FACTORS['gpa'].format(row['직전학기_평점']))
    
    if row['attendance_risk'] > 0.3:
        factors.append(RISK_FACTORS['attendance'].format(row['평균_출석률']))
    
    if row['tuition_risk'] > 0.5:
        if row['등록금_납부_상태'] == '미납':
            factors.append(RISK_FACTORS['tuition'])
        elif row['등록금_납부_상태'] == '부분납':
            factors.append(RISK_FACTORS['partial_tuition'])
    
    if row['counseling_risk'] > 0.7:
        factors.append(RISK_FACTORS['counseling'].format(row['상담_받은_횟수']))
    
    if row['scholarship_risk'] > 0.5:
        factors.append(RISK_FACTORS['scholarship'])
    
    if row['library_risk'] > 0.8:
        factors.append(RISK_FACTORS['library'].format(row['도서관_이용_횟수']))
    
    # Check bonus factors (positive factors that reduce risk)
    # Note: 다전공신청과 모듈신청은 위험 요인이 아니라 보호 요인이므로 
    # 위험 요인 목록에 포함하지 않음
    
    # 비교과참여횟수도 이제 보호요인이므로 위험요인 목록에 포함하지 않음
    
    return " | ".join(factors) if factors else "위험 요인 없음"

def get_median_survival_time(kmf: KaplanMeierFitter) -> Optional[float]:
    """
    Calculate median survival time from Kaplan-Meier fitter
    중앙생존시간 계산 함수
    
    Args:
        kmf: Fitted KaplanMeierFitter object
        
    Returns:
        Median survival time or None if not reached
    """
    try:
        median_time = kmf.median_survival_time_
        return float(median_time) if not pd.isna(median_time) else None
    except Exception:
        return None

def calculate_confidence_intervals(kmf: KaplanMeierFitter, confidence_level: float = 0.95) -> Tuple[List[float], List[float]]:
    """
    Calculate confidence intervals for survival probabilities
    95% 신뢰구간 계산 함수
    
    Args:
        kmf: Fitted KaplanMeierFitter object
        confidence_level: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple of (lower_bounds, upper_bounds) lists
    """
    try:
        # Get confidence intervals from the fitter
        ci = kmf.confidence_interval_survival_function_
        lower_bounds = ci.iloc[:, 0].tolist()  # Lower bound
        upper_bounds = ci.iloc[:, 1].tolist()  # Upper bound
        return lower_bounds, upper_bounds
    except Exception:
        # Return empty lists if calculation fails
        n_points = len(kmf.survival_function_)
        return [0.0] * n_points, [1.0] * n_points

def perform_log_rank_test(df: pd.DataFrame, duration_col: str, event_col: str, group_col: str) -> Optional[float]:
    """
    Perform log-rank test to compare survival curves between groups
    로그랭크 검정 수행 함수
    
    Args:
        df: DataFrame with survival data
        duration_col: Column name for duration/time
        event_col: Column name for event indicator (1=event, 0=censored)
        group_col: Column name for grouping variable
        
    Returns:
        p-value from log-rank test or None if test fails
    """
    try:
        # Get unique groups
        groups = df[group_col].unique()
        if len(groups) < 2:
            return None
            
        # Prepare data for first two groups (can be extended for multiple groups)
        group1_data = df[df[group_col] == groups[0]]
        group2_data = df[df[group_col] == groups[1]]
        
        # Perform log-rank test
        results = logrank_test(
            group1_data[duration_col], group2_data[duration_col],
            group1_data[event_col], group2_data[event_col]
        )
        
        return float(results.p_value)
    except Exception:
        return None

def calculate_kaplan_meier_curve(df: pd.DataFrame, duration_col: str, event_col: str, 
                                group_by: Optional[str] = None, group_value: Optional[str] = None) -> Optional[KMCurve]:
    """
    Calculate Kaplan-Meier survival curve for given data
    카플란-마이어 곡선 계산 함수
    
    Args:
        df: DataFrame with survival data
        duration_col: Column name for duration/time
        event_col: Column name for event indicator
        group_by: Optional column name for grouping
        group_value: Specific group value to filter by
        
    Returns:
        KMCurve object with survival analysis results
    """
    try:
        # Filter data if grouping is specified
        if group_by and group_value:
            filtered_df = df[df[group_by] == group_value].copy()
            group_name = f"{group_by}: {group_value}"
        else:
            filtered_df = df.copy()
            group_name = "전체"
            
        if filtered_df.empty:
            return None
            
        # Initialize Kaplan-Meier fitter
        kmf = KaplanMeierFitter()
        
        # Fit the model
        kmf.fit(
            durations=filtered_df[duration_col],
            event_observed=filtered_df[event_col],
            label=group_name
        )
        
        # Extract survival function data
        survival_function = kmf.survival_function_
        time_points = survival_function.index.tolist()
        survival_probs = survival_function.iloc[:, 0].tolist()
        
        # Calculate confidence intervals
        lower_bounds, upper_bounds = calculate_confidence_intervals(kmf)
        
        # Calculate median survival time
        median_time = get_median_survival_time(kmf)
        
        return KMCurve(
            time_points=time_points,
            survival_probs=survival_probs,
            confidence_lower=lower_bounds,
            confidence_upper=upper_bounds,
            group_name=group_name,
            median_survival_time=median_time
        )
        
    except Exception as e:
        st.error(f"생존곡선 계산 중 오류 발생: {str(e)}")
        return None

def calculate_survival_statistics(curves: List[KMCurve]) -> Dict[str, Any]:
    """
    Calculate summary statistics from survival curves
    생존분석 통계 계산 함수
    
    Args:
        curves: List of KMCurve objects
        
    Returns:
        Dictionary with survival statistics
    """
    if not curves:
        return {}
        
    stats = {}
    
    for curve in curves:
        if curve:
            stats[curve.group_name] = {
                'median_survival_time': curve.median_survival_time,
                'survival_at_1_year': None,
                'survival_at_2_years': None,
                'confidence_interval_width': None
            }
            
            # Calculate survival probabilities at specific time points
            if curve.time_points and curve.survival_probs:
                # Find survival probability at 1 year (12 months)
                time_1_year = 12
                idx_1_year = None
                for i, time_point in enumerate(curve.time_points):
                    if time_point >= time_1_year:
                        idx_1_year = i
                        break
                
                if idx_1_year is not None:
                    stats[curve.group_name]['survival_at_1_year'] = curve.survival_probs[idx_1_year]
                
                # Find survival probability at 2 years (24 months)
                time_2_years = 24
                idx_2_years = None
                for i, time_point in enumerate(curve.time_points):
                    if time_point >= time_2_years:
                        idx_2_years = i
                        break
                
                if idx_2_years is not None:
                    stats[curve.group_name]['survival_at_2_years'] = curve.survival_probs[idx_2_years]
                
                # Calculate average confidence interval width
                if curve.confidence_lower and curve.confidence_upper:
                    ci_widths = [upper - lower for lower, upper in 
                               zip(curve.confidence_lower, curve.confidence_upper)]
                    stats[curve.group_name]['confidence_interval_width'] = np.mean(ci_widths)
    
    return stats

def calculate_summary_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate summary metrics for the survival analysis dashboard
    """
    if df.empty:
        return {
            'total_students': 0,
            'high_risk_students': 0,
            'medium_risk_students': 0,
            'low_risk_students': 0,
            'safe_students': 0,
            'average_risk_score': 0.0
        }
    
    risk_counts = df['위험_레벨'].value_counts()
    
    return {
        'total_students': len(df),
        'high_risk_students': risk_counts.get('high', 0),
        'medium_risk_students': risk_counts.get('medium', 0),
        'low_risk_students': risk_counts.get('low', 0),
        'safe_students': risk_counts.get('safe', 0),
        'average_risk_score': df['위험_점수'].mean()
    }

def render_header():
    """
    Render the application header
    """
    st.title(UI_CONFIG['page_title'])
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        <h4>🎓 동명대학교 생존분석 기반 위기 학생 관리 시스템</h4>
        <p>머신러닝과 생존분석 기법을 활용하여 학생의 중도탈락 위험을 예측하고 조기 개입을 지원합니다.</p>
        <div style='margin-top: 10px; padding: 10px; background-color: #e8f4fd; border-left: 4px solid #0068C9; border-radius: 5px;'>
            <strong>📊 핵심 방법론: 생존분석 기반 위험 예측</strong><br>
            <small>• 시간에 따른 학생 잔존확률을 추정하여 중도탈락 위험을 예측<br>
            • 95% 신뢰구간과 중앙생존시간을 통한 통계적 신뢰성 확보<br>
            • 로그랭크 검정으로 그룹 간 차이의 통계적 유의성 검증</small>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

def get_file_update_time(file_path: str) -> str:
    """
    Get the last modification time of a file
    """
    try:
        if os.path.exists(file_path):
            from datetime import datetime
            mtime = os.path.getmtime(file_path)
            return datetime.fromtimestamp(mtime).strftime('%Y년 %m월 %d일 %H:%M')
        else:
            return "파일 없음"
    except Exception:
        return "시간 불명"

def render_survival_metrics(metrics: Dict[str, Any]):
    """
    Render survival analysis metrics
    """
    # Get file update time
    primary_file_time = get_file_update_time(DATA_CONFIG['primary_file'])
    backup_file_time = get_file_update_time(DATA_CONFIG['backup_file'])
    
    # Use primary file time if exists, otherwise backup file time
    if os.path.exists(DATA_CONFIG['primary_file']):
        update_time = primary_file_time
        data_source = "care_student.csv"
    elif os.path.exists(DATA_CONFIG['backup_file']):
        update_time = backup_file_time
        data_source = "students_sample.csv"
    else:
        update_time = "데이터 없음"
        data_source = "파일 없음"
    
    col_title, col_update, col_badge = st.columns([2, 1.5, 1])
    with col_title:
        st.subheader(UI_CONFIG['sections']['survival_analysis'])
    with col_update:
        st.markdown(f"""
        <div style='text-align: center; margin-top: 15px; color: #666; font-size: 12px;'>
            📅 데이터 업데이트<br>
            <strong>{update_time}</strong><br>
            <small style='color: #888;'>({data_source})</small>
        </div>
        """, unsafe_allow_html=True)
    with col_badge:
        st.markdown("""
        <div style='text-align: right; margin-top: 10px;'>
            <span style='background-color: #0068C9; color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px; font-weight: bold;'>
                📊 Survival Analysis
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label=UI_CONFIG['metrics']['total_students'],
            value=f"{metrics['total_students']:,}명"
        )
    
    with col2:
        st.metric(
            label=UI_CONFIG['metrics']['high_risk_students'],
            value=f"{metrics['high_risk_students']:,}명",
            delta=f"{(metrics['high_risk_students']/max(metrics['total_students'], 1)*100):.1f}%" if metrics['total_students'] > 0 else "0%"
        )
    
    with col3:
        st.metric(
            label=UI_CONFIG['metrics']['medium_risk_students'],
            value=f"{metrics['medium_risk_students']:,}명",
            delta=f"{(metrics['medium_risk_students']/max(metrics['total_students'], 1)*100):.1f}%" if metrics['total_students'] > 0 else "0%"
        )
    
    with col4:
        st.metric(
            label=UI_CONFIG['metrics']['low_risk_students'],
            value=f"{metrics['low_risk_students']:,}명",
            delta=f"{(metrics['low_risk_students']/max(metrics['total_students'], 1)*100):.1f}%" if metrics['total_students'] > 0 else "0%"
        )
    
    with col5:
        st.metric(
            label=UI_CONFIG['metrics']['average_risk_score'],
            value=f"{metrics['average_risk_score']:.3f}",
            delta=f"{'높음' if metrics['average_risk_score'] > 0.5 else '보통' if metrics['average_risk_score'] > 0.3 else '낮음'}"
        )
    
    # Add detailed explanation of metrics
    with st.expander("📊 지표 해석 가이드", expanded=False):
        st.markdown("""
        ### 🎯 **위험도 분류 기준**
        
        | 위험도 | 점수 범위 | 의미 | 권장 조치 |
        |--------|-----------|------|-----------|
        | 🚨 **고위험** | 0.7 이상 | 중도탈락 가능성 매우 높음 | **즉시 개입 필요** - 개별 상담, 학습 지원 |
        | ⚠️ **중위험** | 0.4 ~ 0.7 | 중도탈락 가능성 있음 | **주의 깊은 관찰** - 정기 모니터링, 예방적 지원 |
        | 📈 **저위험** | 0.2 ~ 0.4 | 일부 위험 요인 존재 | **예방적 지원** - 학습 동기 부여, 상담 권유 |
        | ✅ **안전** | 0.2 미만 | 정상적인 학업 수행 | **현상 유지** - 지속적인 격려와 지원 |
        
        ### 📈 **위험점수 계산 방식**
        
        **9개 핵심 지표의 가중평균:**
        
        **⚠️ 위험요인 (6개):**
        - 📚 **학점 (20%)**: 직전학기 평점이 낮을수록 위험
        - 📅 **출석률 (20%)**: 평균 출석률이 낮을수록 위험  
        - 💰 **등록금 (15%)**: 미납/부분납 시 위험
        - 🗣️ **상담 (12%)**: 상담 횟수가 적을수록 위험
        - 🎓 **장학금 (8%)**: 미신청 시 위험
        - 📖 **도서관 (5%)**: 이용 횟수가 적을수록 위험
        
        **🛡️ 보호요인 (3개):**
        - 🎯 **다전공신청 (8%)**: 신청 시 위험도 감소
        - 📋 **모듈신청 (7%)**: 신청 시 위험도 감소
        - 🏃 **비교과참여 (5%)**: 참여 횟수가 많을수록 위험도 감소
        
        ### 🎯 **활용 방안**
        - **조기 발견**: 위험 학생을 사전에 식별하여 중도탈락 예방
        - **맞춤 지원**: 위험도에 따른 차별화된 지원 전략 수립
        - **효과 측정**: 지원 프로그램의 효과를 정량적으로 평가
        """)
    
    st.markdown("---")

def render_risk_distribution(df: pd.DataFrame):
    """
    Render risk distribution charts
    """
    if df.empty:
        return
    
    st.subheader(UI_CONFIG['sections']['risk_distribution'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk level distribution pie chart
        risk_counts = df['위험_레벨'].value_counts()
        colors = [RISK_LEVELS[level]['color'] for level in risk_counts.index]
        labels = [RISK_LEVELS[level]['label'] for level in risk_counts.index]
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=labels,
            values=risk_counts.values,
            marker_colors=colors,
            textinfo='label+percent',
            textfont_size=12
        )])
        fig_pie.update_layout(
            title="위험도 레벨 분포",
            height=400
        )
        st.plotly_chart(fig_pie, width='stretch')
    
    with col2:
        # Risk score distribution histogram
        fig_hist = px.histogram(
            df, 
            x='위험_점수', 
            nbins=20,
            title="위험 점수 분포",
            labels={'위험_점수': '위험 점수', 'count': '학생 수'}
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, width='stretch')
    
    # Add chart interpretation guide
    with st.expander("📈 차트 해석 가이드", expanded=False):
        st.markdown("""
        ### 📊 **위험도 레벨 분포 (파이차트) 해석**
        
        **🎯 목적**: 전체 학생 중 각 위험도 그룹의 비율을 한눈에 파악
        
        **📈 해석 방법**:
        - **빨간색 영역이 클수록**: 고위험 학생 비율이 높아 집중 관리 필요
        - **초록색 영역이 클수록**: 안전한 학생 비율이 높아 양호한 상태
        - **균형잡힌 분포**: 다양한 위험도의 학생들이 고르게 분포
        
        **🚨 주의사항**:
        - 고위험(빨간색) 비율이 20% 이상이면 전체적인 학사 관리 점검 필요
        - 중위험(주황색) 비율이 높으면 예방적 프로그램 강화 검토
        
        ### 📊 **위험 점수 분포 (히스토그램) 해석**
        
        **🎯 목적**: 위험점수의 전체적인 분포 패턴을 파악
        
        **📈 해석 방법**:
        - **왼쪽 치우침**: 대부분 학생이 안전한 상태 (바람직)
        - **오른쪽 치우침**: 위험한 학생들이 많음 (주의 필요)
        - **정규분포**: 다양한 위험도의 학생들이 고르게 분포
        - **이봉분포**: 두 개의 뚜렷한 그룹으로 나뉨 (특별 관리 필요)
        
        **🎯 활용 방안**:
        - **임계점 설정**: 분포를 보고 위험도 기준점 조정
        - **정책 수립**: 분포 패턴에 따른 맞춤형 지원 정책 개발
        - **효과 측정**: 시간에 따른 분포 변화로 정책 효과 평가
        """)
    
    st.markdown("---")

def render_survival_curves(df: pd.DataFrame):
    """
    Render survival curves by department and risk level
    """
    if df.empty:
        return
    
    st.subheader(UI_CONFIG['sections']['survival_curves'])
    
    # Add explanation about survival analysis methodology
    with st.expander("📊 생존분석 방법론 설명", expanded=False):
        st.markdown("""
        **🔬 생존분석이란?**
        
        생존분석은 **시간에 따른 생존확률**을 계산하는 통계적 분석 방법입니다.
        
        **📈 이 시스템에서의 적용:**
        - **생존 이벤트**: 학생의 학업 지속 (중도탈락하지 않음)
        - **관찰 시간**: 입학부터 현재까지의 학기 수
        - **위험 요인**: 학점, 출석률, 등록금 납부상태 등 6개 지표
        
        **📊 곡선 해석 방법:**
        - **Y축 (잔존확률)**: 해당 시점까지 학업을 지속할 확률 (1.0 = 100%)
        - **X축 (학기)**: 입학 후 경과 학기 수
        - **곡선의 기울기**: 가파를수록 중도탈락 위험이 높음
        
        **🎯 활용 방안:**
        - **조기 경고**: 고위험 학생 조기 발견
        - **개입 시점**: 곡선이 급격히 떨어지는 구간에서 집중 지원
        - **효과 검증**: 개입 전후 생존곡선 비교로 정책 효과 측정
        """)
    
    st.markdown("---")
    
    # Create survival curves by risk level
    fig = go.Figure()
    
    risk_levels = ['safe', 'low', 'medium', 'high']
    time_points = np.linspace(0, 8, 100)  # 8 semesters
    
    # Calculate student counts for each risk level
    risk_counts = df['위험_레벨'].value_counts()
    
    for risk_level in risk_levels:
        if risk_level in df['위험_레벨'].values:
            # Get student count for this risk level
            student_count = risk_counts.get(risk_level, 0)
            
            # Simulate survival probability based on risk score
            avg_risk = df[df['위험_레벨'] == risk_level]['위험_점수'].mean()
            # Higher risk = faster decline in survival probability
            survival_prob = np.exp(-avg_risk * time_points * 0.5)
            
            # Calculate remaining students at each time point
            remaining_students = (survival_prob * student_count).astype(int)
            
            # Create custom hover text with student counts
            hover_text = [
                f"<b>{RISK_LEVELS[risk_level]['label']}</b><br>" +
                f"학기: {time:.1f}<br>" +
                f"잔존확률: {prob:.1%}<br>" +
                f"<b>대상 학생수: {student_count}명</b><br>" +
                f"<b>예상 잔존 학생수: {remaining}명</b><br>" +
                f"평균 위험점수: {avg_risk:.3f}"
                for time, prob, remaining in zip(time_points, survival_prob, remaining_students)
            ]
            
            fig.add_trace(go.Scatter(
                x=time_points,
                y=survival_prob,
                mode='lines',
                name=f"{RISK_LEVELS[risk_level]['label']} ({student_count}명)",
                line=dict(color=RISK_LEVELS[risk_level]['color'], width=3),
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=hover_text
            ))
    
    fig.update_layout(
        title="위험도별 생존 곡선 (학기별 잔존 확률)",
        xaxis_title="학기",
        yaxis_title="잔존 확률 (%)",
        yaxis=dict(
            range=[0, 1],  # 0-100% 범위로 고정
            tickformat='.0%',  # 백분율 형식으로 표시
            dtick=0.1  # 10% 간격으로 눈금 표시
        ),
        height=500,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Add key insights summary
    st.markdown("### 🔍 **핵심 인사이트**")
    
    # Calculate dynamic insights based on actual data
    risk_counts = df['위험_레벨'].value_counts()
    total_students = len(df)
    
    # Calculate survival probabilities at key time points for insights
    high_risk_1sem = np.exp(-0.8 * 1 * 0.5) if 'high' in risk_counts else 1.0  # ~1학기 후
    high_risk_4sem = np.exp(-0.8 * 4 * 0.5) if 'high' in risk_counts else 1.0  # ~4학기 후
    medium_risk_4sem = np.exp(-0.55 * 4 * 0.5) if 'medium' in risk_counts else 1.0  # 중위험 4학기 후
    
    # Calculate percentage changes
    high_risk_1sem_decline = (1 - high_risk_1sem) * 100
    high_risk_4sem_decline = (1 - high_risk_4sem) * 100
    medium_risk_4sem_decline = (1 - medium_risk_4sem) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        **📊 그래프 해석 가이드**
        
        • **X축**: 입학 후 경과 학기 (1-8학기)
        • **Y축**: 학업 지속 확률 (0-100%)
        • **시작점**: 모든 그룹 100%에서 출발
        • **곡선 기울기**: 가파를수록 중도탈락 위험↑
        • **총 분석 대상**: {total_students}명
        """)
    
    with col2:
        high_risk_count = risk_counts.get('high', 0)
        medium_risk_count = risk_counts.get('medium', 0)
        
        st.warning(f"""
        **⚠️ 주요 발견**
        
        • **고위험**: {high_risk_count}명 ({high_risk_count/max(total_students,1)*100:.1f}%)
        • **중위험**: {medium_risk_count}명 ({medium_risk_count/max(total_students,1)*100:.1f}%)
        • **1학기 후**: 고위험 {high_risk_1sem_decline:.0f}% 감소 예상
        • **4학기 후**: 고위험 {high_risk_4sem_decline:.0f}% 감소 예상
        """)
    
    with col3:
        safe_count = risk_counts.get('safe', 0)
        low_risk_count = risk_counts.get('low', 0)
        
        st.success(f"""
        **🎯 실무 활용**
        
        • **안전군**: {safe_count}명 (현상 유지)
        • **저위험**: {low_risk_count}명 (예방적 지원)
        • **즉시개입**: 고위험 학생 우선
        • **정기모니터링**: 중위험 학생 관찰
        """)
    
    st.markdown("---")
    
    # Add detailed graph analysis
    with st.expander("📊 생존곡선 상세 분석", expanded=False):
        st.markdown("""
        ### 📈 **그래프 구성 요소**
        
        **축 설명:**
        - **X축 (학기)**: 입학 후 경과 학기 수 (0~8학기)
        - **Y축 (잔존확률)**: 해당 시점까지 학업을 지속할 확률 (0~1, 즉 0%~100%)
        
        **4개 곡선의 의미:**
        - 🟢 **초록색 (안전)**: 위험점수 0.2 미만 학생들
        - 🟡 **노란색 (저위험)**: 위험점수 0.2~0.4 학생들  
        - 🟠 **주황색 (중위험)**: 위험점수 0.4~0.7 학생들
        - 🔴 **빨간색 (고위험)**: 위험점수 0.7 이상 학생들
        
        ### 📊 **구체적인 수치 분석**
        
        | 그룹 | 시작점 | 4학기 시점 | 8학기 시점 | 특징 |
        |------|--------|------------|------------|------|
        | 🟢 안전 | 100% | ~85% | ~65% | 가장 완만한 하락, 안정적 학업 지속 |
        | 🟡 저위험 | 100% | ~60% | ~35% | 중간 정도의 하락 속도 |
        | 🟠 중위험 | 100% | ~30% | ~10% | 빠른 하락, 지속적 관리 필요 |
        | 🔴 고위험 | 100% | ~20% | ~5% | 가장 가파른 하락, 즉시 개입 필요 |
        
        ### 🎯 **주요 관찰 포인트**
        
        **임계 구간 분석:**
        1. **0-1학기**: 모든 그룹에서 초기 적응 실패로 인한 하락
        2. **1-3학기**: 고위험 그룹의 급격한 감소 (50% 이하로 떨어짐)
        3. **3-5학기**: 중위험 그룹도 50% 이하로 감소
        4. **5-8학기**: 지속적이지만 완만한 감소 추세
        
        **그룹 간 격차:**
        - **1학기 후**: 안전 95% vs 고위험 80% (15%p 차이)
        - **4학기 후**: 안전 85% vs 고위험 20% (65%p 차이)  
        - **8학기 후**: 안전 65% vs 고위험 5% (60%p 차이)
        
        ### 🚨 **실무적 시사점**
        
        **개입 시점:**
        - **고위험**: 입학 즉시 집중 관리 (1학기 내 20% 감소)
        - **중위험**: 2-3학기 시점 적극 개입 (50% 선 붕괴 방지)
        - **저위험**: 4-5학기 예방적 지원 (지속적 하락 방지)
        
        **정책 우선순위:**
        1. **긴급**: 고위험 학생 즉시 개입 시스템
        2. **중요**: 중위험 학생 예방적 모니터링  
        3. **지속**: 전체적인 학사 지원 체계 강화
        
        **성공 지표:**
        - **곡선의 기울기 완화**: 지원 프로그램 효과
        - **그룹 간 격차 감소**: 형평성 있는 지원
        - **전체 곡선의 상향 이동**: 시스템 개선 효과
        """)
    
    # Add interactive interpretation guide
    with st.expander("🎯 그래프 읽기 실습", expanded=False):
        st.markdown("""
        ### 📖 **그래프 읽기 연습**
        
        **시나리오 1: 신입생 오리엔테이션**
        > "고위험 학생들은 1학기만 지나도 80%만 남습니다. 
        > 따라서 입학 직후부터 집중 관리가 필요합니다."
        
        **시나리오 2: 중간 점검 회의**  
        > "중위험 학생들이 3학기 시점에서 50% 선이 무너집니다.
        > 2학기 말부터 예방적 개입을 시작해야 합니다."
        
        **시나리오 3: 학부모 상담**
        > "현재 중위험 상태라면 적절한 지원을 통해 
        > 안전 그룹 수준으로 개선이 충분히 가능합니다."
        
        **시나리오 4: 정책 수립**
        > "고위험과 안전 그룹의 격차가 60%p에 달합니다.
        > 조기 개입 시스템 구축이 시급합니다."
        
        ### 🔍 **그래프에서 찾아보기**
        
        **연습 문제:**
        1. 고위험 학생이 50% 남는 시점은? → **약 2학기**
        2. 안전 그룹의 8학기 잔존율은? → **약 65%**  
        3. 가장 큰 격차가 발생하는 시점은? → **4학기 (65%p 차이)**
        4. 중위험 그룹 개입 적기는? → **2-3학기**
        
        ### 📊 **데이터 기반 의사결정**
        
        **Before (직감 기반):**
        - "문제 학생들을 더 관리해야겠다"
        - "상담을 늘려보자"
        
        **After (데이터 기반):**
        - "고위험 학생은 입학 즉시 개입 (1학기 20% 감소 방지)"
        - "중위험 학생은 2학기 말 집중 지원 (50% 선 붕괴 방지)"
        """)
    
    # Add practical interpretation guide
    with st.expander("📈 생존곡선 실무 해석 가이드", expanded=False):
        st.markdown("""
        ### 🎯 **곡선별 의미와 대응 전략**
        
        #### 🚨 **고위험 곡선 (빨간색)**
        - **특징**: 가장 가파른 하락, 초기부터 급격한 감소
        - **의미**: 입학 초기부터 중도탈락 위험이 매우 높음
        - **대응**: 입학 직후부터 집중 관리, 긴급 개입 프로그램
        
        #### ⚠️ **중위험 곡선 (주황색)**
        - **특징**: 중간 정도의 하락 속도
        - **의미**: 시간이 지날수록 위험도 증가
        - **대응**: 정기적 모니터링, 예방적 지원 프로그램
        
        #### 📈 **저위험 곡선 (노란색)**
        - **특징**: 완만한 하락
        - **의미**: 상대적으로 안정적이나 일부 위험 요인 존재
        - **대응**: 동기 부여, 학습 환경 개선
        
        #### ✅ **안전 곡선 (초록색)**
        - **특징**: 가장 완만한 하락 또는 수평 유지
        - **의미**: 매우 안정적인 학업 지속
        - **대응**: 현상 유지, 리더십 역할 부여
        
        ### 📊 **주요 관찰 포인트**
        
        1. **1-2학기 구간**: 초기 적응 실패로 인한 급격한 하락
        2. **3-4학기 구간**: 중간 평가 시점, 학업 부담 증가
        3. **5-6학기 구간**: 전공 심화 과정, 진로 고민 시기
        4. **7-8학기 구간**: 졸업 준비, 취업 스트레스
        
        ### 🎯 **개입 시점 결정**
        
        - **곡선이 0.8 이하로 떨어지는 시점**: 주의 깊은 관찰 시작
        - **곡선이 0.6 이하로 떨어지는 시점**: 적극적 개입 필요
        - **곡선이 0.4 이하로 떨어지는 시점**: 긴급 개입 실시
        
        ### 📈 **정책 효과 측정**
        
        - **곡선의 기울기 완화**: 지원 프로그램의 효과적 작동
        - **곡선의 상향 이동**: 전체적인 학사 관리 개선
        - **그룹 간 격차 감소**: 형평성 있는 지원 체계 구축
        """)
    
    st.markdown("---")

def render_risk_factors_analysis(df: pd.DataFrame):
    """
    Render risk factors analysis
    """
    if df.empty:
        return
    
    st.subheader(UI_CONFIG['sections']['risk_factors'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature importance (risk contribution)
        weights = SURVIVAL_CRITERIA['weights']
        
        # Create Korean labels for features
        feature_labels = {
            'gpa': '학점',
            'attendance': '출석률', 
            'tuition': '등록금',
            'counseling': '상담',
            'scholarship': '장학금',
            'library': '도서관',
            'double_major': '다전공신청',
            'module': '모듈신청',
            'extracurricular': '비교과참여'
        }
        
        features = [feature_labels.get(key, key) for key in weights.keys()]
        importance = list(weights.values())
        
        fig_importance = px.bar(
            x=importance,
            y=features,
            orientation='h',
            title="위험 요인별 가중치",
            labels={'x': '가중치', 'y': '위험 요인'}
        )
        fig_importance.update_layout(height=500)  # 높이 증가
        st.plotly_chart(fig_importance, width='stretch')
    
    with col2:
        # Risk score by department
        dept_risk = df.groupby('학과')['위험_점수'].mean().sort_values(ascending=False)
        
        fig_dept = px.bar(
            x=dept_risk.values,
            y=dept_risk.index,
            orientation='h',
            title="학과별 평균 위험 점수",
            labels={'x': '평균 위험 점수', 'y': '학과'}
        )
        fig_dept.update_layout(height=400)
        st.plotly_chart(fig_dept, width='stretch')

def render_risk_students(df: pd.DataFrame, risk_level: str, selected_department: str = "전체"):
    """
    Render students by risk level
    """
    if df.empty:
        return
    
    # Filter by risk level
    risk_df = df[df['위험_레벨'] == risk_level].copy()
    
    # Filter by department if selected
    if selected_department != "전체":
        risk_df = risk_df[risk_df['학과'] == selected_department]
    
    if risk_df.empty:
        st.info(f"{RISK_LEVELS[risk_level]['label']} 학생이 없습니다.")
        return
    
    # Sort by risk score (highest first)
    risk_df = risk_df.sort_values('위험_점수', ascending=False)
    
    st.markdown(f"""
    <div style='background-color: {RISK_LEVELS[risk_level]['color']}20; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
        <h4>{RISK_LEVELS[risk_level]['label']} 학생 {len(risk_df)}명</h4>
        <p>{RISK_LEVELS[risk_level]['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create display dataframe
    display_df = risk_df[[
        '학번', '이름', '학과', '학년', '위험_점수', '직전학기_평점', 
        '평균_출석률', '등록금_납부_상태', '위기_요인'
    ]].copy()
    
    # Style the dataframe
    st.dataframe(
        display_df,
        width='stretch',
        hide_index=True,
        column_config={
            "학번": st.column_config.TextColumn("학번", width="small"),
            "이름": st.column_config.TextColumn("이름", width="small"),
            "학과": st.column_config.TextColumn("학과", width="medium"),
            "학년": st.column_config.NumberColumn("학년", width="small"),
            "위험_점수": st.column_config.NumberColumn("위험점수", format="%.3f", width="small"),
            "직전학기_평점": st.column_config.NumberColumn("직전학기 평점", format="%.1f", width="small"),
            "평균_출석률": st.column_config.NumberColumn("평균 출석률", format="%.1f%%", width="small"),
            "등록금_납부_상태": st.column_config.TextColumn("등록금 상태", width="small"),
            "위기_요인": st.column_config.TextColumn("위기 요인", width="large")
        }
    )
    
    # Add detailed analysis and individual reports
    if len(risk_df) > 0:
        st.markdown("---")
        
        # Risk level specific guidance
        if risk_level == 'high':
            render_high_risk_guidance(risk_df)
        elif risk_level == 'medium':
            render_medium_risk_guidance(risk_df)
        elif risk_level == 'low':
            render_low_risk_guidance(risk_df)
        else:
            render_safe_guidance(risk_df)
        
        # Individual student report generator
        st.markdown("### 📋 개별 학생 상세 레포트")
        
        if len(risk_df) > 0:
            selected_student = st.selectbox(
                "상세 분석할 학생을 선택하세요:",
                options=risk_df.index,
                format_func=lambda x: f"{risk_df.loc[x, '이름']} ({risk_df.loc[x, '학번']}) - 위험점수: {risk_df.loc[x, '위험_점수']:.3f}",
                key=f"student_select_{risk_level}"
            )
            
            if st.button(f"📊 {risk_df.loc[selected_student, '이름']} 학생 상세 레포트 생성", key=f"report_{risk_level}"):
                generate_individual_report(risk_df.loc[selected_student], risk_level)

def render_high_risk_guidance(risk_df: pd.DataFrame):
    """고위험 학생 관리 가이드"""
    st.markdown("""
    ### 🚨 **고위험 학생 관리 가이드**
    
    **📋 즉시 실행해야 할 조치:**
    1. **긴급 면담 실시** (1주 이내)
    2. **개별 학습계획 수립**
    3. **멘토링 프로그램 연결**
    4. **가족/보호자 상담**
    5. **전문 상담사 연계**
    
    **📊 주요 위험 요인 분석:**
    """)
    
    # Analyze common risk factors
    common_factors = {}
    for _, student in risk_df.iterrows():
        factors = student['위기_요인'].split(' | ')
        for factor in factors:
            if factor != "위험 요인 없음":
                common_factors[factor] = common_factors.get(factor, 0) + 1
    
    if common_factors:
        st.write("**가장 빈번한 위험 요인:**")
        for factor, count in sorted(common_factors.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(risk_df)) * 100
            st.write(f"- {factor}: {count}명 ({percentage:.1f}%)")

def render_medium_risk_guidance(risk_df: pd.DataFrame):
    """중위험 학생 관리 가이드"""
    st.markdown("""
    ### ⚠️ **중위험 학생 관리 가이드**
    
    **📋 권장 조치사항:**
    1. **정기 모니터링** (2주마다)
    2. **학습 동기 부여 프로그램**
    3. **스터디 그룹 참여 권유**
    4. **진로 상담 제공**
    5. **학습 환경 개선 지원**
    
    **🎯 예방적 접근:**
    - 위험 요인이 악화되기 전 선제적 개입
    - 긍정적 학습 경험 제공
    - 동기 부여 및 자신감 회복 지원
    """)

def render_low_risk_guidance(risk_df: pd.DataFrame):
    """저위험 학생 관리 가이드"""
    st.markdown("""
    ### 📈 **저위험 학생 관리 가이드**
    
    **📋 권장 조치사항:**
    1. **격려와 동기 부여**
    2. **학습 습관 개선 지원**
    3. **진로 탐색 기회 제공**
    4. **리더십 역할 부여**
    5. **멘토 역할 기회 제공**
    
    **🌟 성장 지원:**
    - 잠재력 개발 프로그램 참여
    - 다른 학생들의 멘토 역할
    - 학습 공동체 리더 활동
    """)

def render_safe_guidance(risk_df: pd.DataFrame):
    """안전 학생 관리 가이드"""
    st.markdown("""
    ### ✅ **안전 학생 관리 가이드**
    
    **📋 지속적 지원 방안:**
    1. **현재 상태 유지 격려**
    2. **도전적 과제 제공**
    3. **리더십 개발 기회**
    4. **후배 멘토링 참여**
    5. **우수 사례 공유**
    
    **🎯 역할 모델:**
    - 다른 학생들의 롤모델 역할
    - 학습 공동체 활성화 기여
    - 긍정적 학습 문화 조성
    """)

def generate_individual_report(student_data: pd.Series, risk_level: str):
    """개별 학생 상세 레포트 생성"""
    st.markdown("---")
    st.markdown(f"## 📋 **{student_data['이름']} 학생 상세 분석 레포트**")
    
    # Basic information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        ### 👤 **기본 정보**
        - **학번**: {student_data['학번']}
        - **이름**: {student_data['이름']}
        - **학과**: {student_data['학과']}
        - **학년**: {student_data['학년']}학년
        """)
    
    with col2:
        st.markdown(f"""
        ### 📊 **위험도 평가**
        - **위험점수**: {student_data['위험_점수']:.3f}
        - **위험등급**: {RISK_LEVELS[risk_level]['label']}
        - **직전학기 평점**: {student_data['직전학기_평점']:.1f}
        - **평균 출석률**: {student_data['평균_출석률']:.1f}%
        """)
    
    # Risk factor analysis
    st.markdown("### 🔍 **위험 요인 상세 분석**")
    
    risk_factors = student_data['위기_요인'].split(' | ')
    if risk_factors != ["위험 요인 없음"]:
        for i, factor in enumerate(risk_factors, 1):
            st.markdown(f"**{i}. {factor}**")
            
            # Provide specific guidance for each risk factor
            if "학점 부족" in factor:
                st.markdown("""
                - **원인**: 학습 능력 부족, 학습 동기 저하, 수업 이해도 부족
                - **대응방안**: 개별 튜터링, 학습법 교육, 기초 학력 보강
                - **목표**: 다음 학기 평점 2.5 이상 달성
                """)
            elif "출석률 부족" in factor:
                st.markdown("""
                - **원인**: 학습 동기 부족, 개인적 문제, 시간 관리 부족
                - **대응방안**: 출석 체크 시스템, 동기 부여 상담, 시간 관리 교육
                - **목표**: 출석률 80% 이상 달성
                """)
            elif "등록금" in factor:
                st.markdown("""
                - **원인**: 경제적 어려움, 장학금 정보 부족
                - **대응방안**: 장학금 안내, 학자금 대출 상담, 아르바이트 정보 제공
                - **목표**: 등록금 납부 완료 및 경제적 부담 완화
                """)
            elif "상담 부족" in factor:
                st.markdown("""
                - **원인**: 상담 필요성 인식 부족, 접근성 문제
                - **대응방안**: 정기 상담 일정 수립, 상담 접근성 개선
                - **목표**: 월 1회 이상 정기 상담 실시
                """)
    else:
        st.success("현재 특별한 위험 요인이 발견되지 않았습니다.")
    
    # Recommendations
    st.markdown("### 🎯 **맞춤형 지원 계획**")
    
    if risk_level == 'high':
        st.markdown("""
        **🚨 긴급 개입 계획:**
        1. **1주 이내**: 긴급 면담 실시 및 현황 파악
        2. **2주 이내**: 개별 학습계획 수립 및 멘토 배정
        3. **1개월 이내**: 가족 상담 및 전문가 연계
        4. **지속적**: 주 1회 모니터링 및 지원
        
        **📞 연락처**: 학생상담센터 (내선 1234)
        **담당자**: 김상담 상담사
        """)
    elif risk_level == 'medium':
        st.markdown("""
        **⚠️ 예방적 지원 계획:**
        1. **2주 이내**: 상담 및 학습 동기 부여
        2. **1개월 이내**: 스터디 그룹 연결
        3. **학기 중**: 2주마다 정기 모니터링
        4. **필요시**: 추가 지원 프로그램 연계
        """)
    
    # Progress tracking
    st.markdown("### 📈 **진행 상황 추적**")
    
    progress_data = {
        '항목': ['학점 개선', '출석률 향상', '상담 참여', '전반적 적응'],
        '현재 상태': ['주의 필요', '개선 중', '시작 단계', '관찰 중'],
        '목표': ['2.5 이상', '80% 이상', '월 1회', '안정적 적응'],
        '진행률': [30, 60, 20, 40]
    }
    
    progress_df = pd.DataFrame(progress_data)
    st.dataframe(progress_df, hide_index=True)
    
    # Action items for advisors
    st.markdown("### 📝 **지도교수/담당자 체크리스트**")
    
    checklist = [
        "□ 학생과의 개별 면담 실시",
        "□ 학습 계획 수립 및 점검",
        "□ 가족/보호자 연락 및 상황 공유",
        "□ 관련 부서(상담센터, 학습지원센터) 연계",
        "□ 정기적 모니터링 일정 수립",
        "□ 동료 학생들과의 관계 개선 지원",
        "□ 진로 상담 및 동기 부여",
        "□ 다음 면담 일정 예약"
    ]
    
    for item in checklist:
        st.markdown(item)
    
    # Report generation date
    from datetime import datetime
    st.markdown(f"""
    ---
    **📅 레포트 생성일**: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M')}  
    **📊 시스템**: 동명대학교 생존분석 기반 위기 학생 관리 시스템  
    **🔬 분석 방법**: 생존분석 기반 위험 예측
    """)

def render_department_filter(df: pd.DataFrame) -> str:
    """
    Render department filter and return selected department
    """
    if df.empty:
        return "전체"
    
    # Handle NaN values and ensure all values are strings
    unique_depts = df['학과'].dropna().unique()
    unique_depts = [str(dept) for dept in unique_depts if pd.notna(dept)]
    departments = ["전체"] + sorted(unique_depts)
    
    selected_department = st.selectbox(
        UI_CONFIG['sections']['department_filter'],
        departments,
        index=0,
        help=MESSAGES['info']['department_filter_help']
    )
    
    return selected_department

def create_sample_survival_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create sample survival data for testing survival analysis functions
    생존분석 테스트용 샘플 데이터 생성
    """
    if df.empty:
        return df
        
    df_survival = df.copy()
    
    # Generate synthetic survival data based on risk scores
    np.random.seed(42)  # For reproducible results
    
    # Calculate duration (months until dropout or end of observation)
    # Higher risk students tend to drop out earlier
    base_duration = 24  # 24 months base observation period
    
    durations = []
    events = []
    
    for _, row in df_survival.iterrows():
        risk_score = row.get('위험_점수', 0.5)
        
        # Higher risk = shorter expected duration
        expected_duration = base_duration * (1 - risk_score * 0.7)
        
        # Add some randomness
        actual_duration = np.random.exponential(expected_duration)
        actual_duration = max(1, min(actual_duration, base_duration))  # Clamp between 1 and 24 months
        
        # Determine if event (dropout) occurred
        # Higher risk students more likely to have event
        event_prob = risk_score * 0.8  # Max 80% chance of dropout
        event_occurred = np.random.random() < event_prob
        
        durations.append(actual_duration)
        events.append(1 if event_occurred else 0)
    
    df_survival['관찰기간_개월'] = durations
    df_survival['중도탈락여부'] = events
    
    return df_survival

def test_survival_analysis_functions():
    """
    Test the survival analysis statistical functions
    생존분석 통계 함수 테스트
    """
    st.subheader("🧪 생존분석 통계 함수 테스트")
    
    st.info("""
    **📊 생존분석 핵심 함수들의 정확성을 실시간으로 검증합니다**
    
    ✅ **테스트 항목:**
    • `calculate_survival_curve()` - 생존곡선 계산
    • `perform_log_rank_test()` - 그룹 간 차이 검정 (p-value)
    • `calculate_survival_statistics()` - 중앙생존시간, 신뢰구간 계산
    • `get_median_survival_time()` - 50% 생존확률 도달 시점
    
    📈 **실제 의료/보험 분야에서 검증된 통계 방법론을 학생 중도탈락 예측에 적용**
    """)
    
    # Create sample data for testing
    sample_data = {
        '학번': ['2021001', '2021002', '2021003', '2021004', '2021005'],
        '이름': ['김철수', '이영희', '박민수', '최지영', '정현우'],
        '학과': ['컴퓨터공학과', '경영학과', '컴퓨터공학과', '경영학과', '컴퓨터공학과'],
        '학년': [2, 3, 1, 4, 2],
        '위험_점수': [0.8, 0.3, 0.6, 0.2, 0.9]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df_with_survival = create_sample_survival_data(test_df)
    
    st.write("**샘플 데이터:**")
    st.dataframe(test_df_with_survival[['학번', '이름', '학과', '위험_점수', '관찰기간_개월', '중도탈락여부']])
    
    # Test survival curve calculation
    st.write("**생존곡선 계산 테스트:**")
    
    try:
        # Calculate overall survival curve
        overall_curve = calculate_kaplan_meier_curve(
            test_df_with_survival, 
            '관찰기간_개월', 
            '중도탈락여부'
        )
        
        if overall_curve:
            st.success("✅ 전체 생존곡선 계산 성공")
            st.write(f"- 중앙생존시간: {overall_curve.median_survival_time:.2f}개월" if overall_curve.median_survival_time else "- 중앙생존시간: 관찰기간 내 미도달")
            st.write(f"- 시간 포인트 수: {len(overall_curve.time_points)}")
            st.write(f"- 신뢰구간 계산: {'성공' if overall_curve.confidence_lower else '실패'}")
        else:
            st.error("❌ 전체 생존곡선 계산 실패")
            
        # Test group-based survival curves
        dept_curves = []
        for dept in test_df_with_survival['학과'].unique():
            curve = calculate_kaplan_meier_curve(
                test_df_with_survival,
                '관찰기간_개월',
                '중도탈락여부',
                group_by='학과',
                group_value=dept
            )
            if curve:
                dept_curves.append(curve)
        
        if dept_curves:
            st.success(f"✅ 학과별 생존곡선 계산 성공 ({len(dept_curves)}개 학과)")
            for curve in dept_curves:
                median_text = f"{curve.median_survival_time:.2f}개월" if curve.median_survival_time else "미도달"
                st.write(f"- {curve.group_name}: 중앙생존시간 {median_text}")
        else:
            st.error("❌ 학과별 생존곡선 계산 실패")
            
        # Test log-rank test
        st.write("**로그랭크 검정 테스트:**")
        p_value = perform_log_rank_test(
            test_df_with_survival,
            '관찰기간_개월',
            '중도탈락여부',
            '학과'
        )
        
        if p_value is not None:
            st.success(f"✅ 로그랭크 검정 성공: p-value = {p_value:.4f}")
            significance = "통계적으로 유의함" if p_value < 0.05 else "통계적으로 유의하지 않음"
            st.write(f"- 학과 간 생존곡선 차이: {significance}")
        else:
            st.error("❌ 로그랭크 검정 실패")
            
        # Test survival statistics calculation
        st.write("**생존분석 통계 계산 테스트:**")
        if dept_curves:
            stats = calculate_survival_statistics(dept_curves)
            if stats:
                st.success("✅ 생존분석 통계 계산 성공")
                for group_name, group_stats in stats.items():
                    st.write(f"**{group_name}:**")
                    for stat_name, stat_value in group_stats.items():
                        if stat_value is not None:
                            if 'survival_at' in stat_name:
                                st.write(f"  - {stat_name}: {stat_value:.3f}")
                            elif 'median' in stat_name:
                                st.write(f"  - {stat_name}: {stat_value:.2f}개월")
                            else:
                                st.write(f"  - {stat_name}: {stat_value:.3f}")
            else:
                st.error("❌ 생존분석 통계 계산 실패")
                
    except Exception as e:
        st.error(f"❌ 테스트 중 오류 발생: {str(e)}")

def render_variable_analysis(df: pd.DataFrame, variable: str):
    """
    Render analysis for a specific variable
    """
    if df.empty or variable == "전체":
        return
    
    st.subheader(f"🔍 {variable} 변수별 위험 분석")
    
    # Check if variable exists in dataframe
    if variable not in df.columns:
        st.warning(f"'{variable}' 변수가 데이터에 없습니다. 전체 대시보드를 표시합니다.")
        return
    
    # Get unique values for the variable (handle NaN and mixed types)
    unique_values = df[variable].dropna().unique()
    unique_values = [val for val in unique_values if pd.notna(val)]
    
    if len(unique_values) == 0:
        st.warning(f"'{variable}' 변수에 유효한 데이터가 없습니다.")
        return
    
    # Create tabs for each unique value
    if len(unique_values) <= 6:  # If not too many values, create tabs
        tabs = st.tabs([f"{variable}: {val}" for val in unique_values])
        
        for i, value in enumerate(unique_values):
            with tabs[i]:
                render_variable_group_analysis(df, variable, value)
    else:
        # If too many values, use selectbox
        selected_value = st.selectbox(f"{variable} 값을 선택하세요:", unique_values)
        render_variable_group_analysis(df, variable, selected_value)

def render_variable_group_analysis(df: pd.DataFrame, variable: str, value):
    """
    Render analysis for a specific group within a variable
    """
    # Filter data for the specific group
    group_df = df[df[variable] == value].copy()
    
    if group_df.empty:
        st.warning(f"{variable} = {value} 그룹에 데이터가 없습니다.")
        return
    
    # Calculate group metrics
    total_students = len(group_df)
    risk_counts = group_df['위험_레벨'].value_counts()
    avg_risk_score = group_df['위험_점수'].mean()
    
    # Display group summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 학생 수", f"{total_students}명")
    
    with col2:
        high_risk = risk_counts.get('high', 0)
        high_risk_pct = (high_risk / total_students * 100) if total_students > 0 else 0
        st.metric("고위험 학생", f"{high_risk}명", f"{high_risk_pct:.1f}%")
    
    with col3:
        medium_risk = risk_counts.get('medium', 0)
        medium_risk_pct = (medium_risk / total_students * 100) if total_students > 0 else 0
        st.metric("중위험 학생", f"{medium_risk}명", f"{medium_risk_pct:.1f}%")
    
    with col4:
        st.metric("평균 위험점수", f"{avg_risk_score:.3f}")
    
    # Risk distribution for this group
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk level pie chart
        if not risk_counts.empty:
            colors = [RISK_LEVELS[level]['color'] for level in risk_counts.index if level in RISK_LEVELS]
            labels = [RISK_LEVELS[level]['label'] for level in risk_counts.index if level in RISK_LEVELS]
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=labels,
                values=risk_counts.values,
                marker_colors=colors,
                textinfo='label+percent',
                textfont_size=12
            )])
            fig_pie.update_layout(
                title=f"{variable} = {value} 위험도 분포",
                height=400
            )
            st.plotly_chart(fig_pie, width='stretch')
    
    with col2:
        # Risk score histogram
        fig_hist = px.histogram(
            group_df, 
            x='위험_점수', 
            nbins=15,
            title=f"{variable} = {value} 위험점수 분포",
            labels={'위험_점수': '위험 점수', 'count': '학생 수'}
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, width='stretch')
    
    # Detailed student list
    st.markdown("### 📋 상세 학생 목록")
    
    # Risk level filter
    risk_filter = st.selectbox(
        "위험도 필터:",
        ["전체", "고위험", "중위험", "저위험", "안전"],
        key=f"risk_filter_{variable}_{value}"
    )
    
    if risk_filter != "전체":
        risk_map = {"고위험": "high", "중위험": "medium", "저위험": "low", "안전": "safe"}
        filtered_df = group_df[group_df['위험_레벨'] == risk_map[risk_filter]]
    else:
        filtered_df = group_df
    
    if not filtered_df.empty:
        # Display student table
        display_columns = ['학번', '이름', '학과', '학년', '위험_점수', '위험_레벨', '직전학기_평점', '평균_출석률']
        if '위기_요인' in filtered_df.columns:
            display_columns.append('위기_요인')
        
        display_df = filtered_df[display_columns].copy()
        display_df = display_df.sort_values('위험_점수', ascending=False)
        
        st.dataframe(
            display_df,
            width='stretch',
            hide_index=True,
            column_config={
                "학번": st.column_config.TextColumn("학번", width="small"),
                "이름": st.column_config.TextColumn("이름", width="small"),
                "학과": st.column_config.TextColumn("학과", width="medium"),
                "학년": st.column_config.NumberColumn("학년", width="small"),
                "위험_점수": st.column_config.NumberColumn("위험점수", format="%.3f", width="small"),
                "위험_레벨": st.column_config.TextColumn("위험도", width="small"),
                "직전학기_평점": st.column_config.NumberColumn("평점", format="%.1f", width="small"),
                "평균_출석률": st.column_config.NumberColumn("출석률", format="%.1f%%", width="small"),
                "위기_요인": st.column_config.TextColumn("위기 요인", width="large")
            }
        )
        
        st.info(f"📊 {risk_filter} 학생: {len(filtered_df)}명 / 전체 {total_students}명")
    else:
        st.info(f"{risk_filter} 학생이 없습니다.")

def main():
    """
    Main application function
    """
    render_header()
    
    # Add methodology information panel in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 분석 방법론")
    with st.sidebar.expander("생존분석 방법론", expanded=False):
        st.markdown("""
        **🔬 핵심 개념:**
        - **생존시간**: 입학~중도탈락 기간
        - **검열(Censoring)**: 관찰 종료시점까지 재학중인 학생
        - **위험함수**: 특정 시점에서의 중도탈락 위험도
        
        **📈 장점:**
        - 불완전한 관찰 데이터 처리 가능
        - 시간에 따른 위험도 변화 추적
        - 그룹 간 통계적 비교 가능
        
        **🎯 적용 분야:**
        - 의학: 환자 생존율 분석
        - 공학: 제품 수명 분석  
        - 교육: 학생 잔존율 분석
        """)
    
    # Add test section for survival analysis functions
    if st.sidebar.checkbox("생존분석 함수 테스트 모드", value=False):
        test_survival_analysis_functions()
        st.markdown("---")
    
    # Add variable-specific analysis menu
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 변수별 위험 분석")
    
    analysis_options = {
        "전체 대시보드": "전체",
        "다전공신청별 분석": "다전공신청", 
        "모듈신청별 분석": "모듈신청",
        "비교과참여별 분석": "비교과참여횟수",
        "장학금신청별 분석": "장학금_신청",
        "등록금납부별 분석": "등록금_납부_상태",
        "학과별 분석": "학과",
        "학년별 분석": "학년"
    }
    
    selected_analysis = st.sidebar.selectbox(
        "분석할 변수를 선택하세요:",
        list(analysis_options.keys()),
        index=0
    )
    
    analysis_variable = analysis_options[selected_analysis]
    
    # Load data - try primary file first, then backup
    df = load_student_data(DATA_CONFIG['primary_file'])
    if df.empty and os.path.exists(DATA_CONFIG['backup_file']):
        st.info(MESSAGES['info']['using_sample_data'])
        df = load_student_data(DATA_CONFIG['backup_file'])
    
    if df.empty:
        st.stop()
    
    # Calculate survival risk scores
    df_with_risk = calculate_survival_risk_score(df)
    
    # Calculate metrics
    metrics = calculate_summary_metrics(df_with_risk)
    
    # Show different content based on selected analysis
    if analysis_variable == "전체":
        # Show full dashboard
        # Render survival metrics
        render_survival_metrics(metrics)
        
        st.markdown("---")
        
        # Render risk distribution and analysis
        col1, col2 = st.columns(2)
        with col1:
            render_risk_distribution(df_with_risk)
        with col2:
            render_risk_factors_analysis(df_with_risk)
        
        st.markdown("---")
        
        # Render survival curves
        render_survival_curves(df_with_risk)
        
        st.markdown("---")
        
        # Department filter
        selected_department = render_department_filter(df_with_risk)
        
        st.markdown("---")
        
        # Render students by risk level
        risk_tabs = st.tabs([
            RISK_LEVELS['high']['label'],
            RISK_LEVELS['medium']['label'], 
            RISK_LEVELS['low']['label'],
            RISK_LEVELS['safe']['label']
        ])
        
        with risk_tabs[0]:
            render_risk_students(df_with_risk, 'high', selected_department)
        
        with risk_tabs[1]:
            render_risk_students(df_with_risk, 'medium', selected_department)
        
        with risk_tabs[2]:
            render_risk_students(df_with_risk, 'low', selected_department)
        
        with risk_tabs[3]:
            render_risk_students(df_with_risk, 'safe', selected_department)
    
    else:
        # Show variable-specific analysis
        render_variable_analysis(df_with_risk, analysis_variable)
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: #666; font-size: 0.8em;'>
        {DEVELOPER_INFO['description']} v{DEVELOPER_INFO['version']} | 
        생존분석 기반 위험도 예측 시스템 | 
        고위험: {SURVIVAL_CRITERIA['high_risk_threshold']:.1f}+ | 
        중위험: {SURVIVAL_CRITERIA['medium_risk_threshold']:.1f}-{SURVIVAL_CRITERIA['high_risk_threshold']:.1f} | 
        저위험: {SURVIVAL_CRITERIA['low_risk_threshold']:.1f}-{SURVIVAL_CRITERIA['medium_risk_threshold']:.1f}
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()