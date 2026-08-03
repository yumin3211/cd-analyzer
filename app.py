import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 웹사이트 기본 설정
st.set_page_config(page_title="Chirality Analyzer", layout="wide")
st.title("📊 CD 스펙트럼 대칭성 자동 분석기")
st.write("농도 오차(진폭)를 자동으로 정규화(Normalization)하고, 수학적 대칭성 점수와 핵심 피크를 자동 진단합니다.")

# 1. 파일 업로드 칸 만들기
col1, col2 = st.columns(2)
with col1:
    file_r = st.file_uploader("R-form CSV 파일을 올려주세요", type=['csv'])
with col2:
    file_s = st.file_uploader("S-form CSV 파일을 올려주세요", type=['csv'])

# 2. 파일 2개가 모두 올라오면 자동 분석 실행
if file_r and file_s:
    # CSV 데이터 전처리 (상단 21줄 스킵 및 숫자 데이터 변환)
    df_r = pd.read_csv(file_r, skiprows=21, header=None)
    df_s = pd.read_csv(file_s, skiprows=21, header=None)
    
    df_r[0] = pd.to_numeric(df_r[0], errors='coerce')
    df_r[2] = pd.to_numeric(df_r[2], errors='coerce')
    df_r = df_r.dropna(subset=[0, 2])
    
    df_s[0] = pd.to_numeric(df_s[0], errors='coerce')
    df_s[2] = pd.to_numeric(df_s[2], errors='coerce')
    df_s = df_s.dropna(subset=[0, 2])
    
    # 데이터 정규화 (최대 피크 높이를 1로 스케일링)
    r_norm = df_r[2] / df_r[2].abs().max()
    s_norm = df_s[2] / df_s[2].abs().max()
    
    # 수학적 대칭성 점수(상관계수) 계산
    try:
        similarity = np.corrcoef(r_norm, -s_norm)[0, 1] * 100
    except:
        similarity = 0
    
    st.divider() 
    
    # [결과 1] 대칭성 점수 및 AI 원인 분석 진단 박스
    st.subheader(f"✅ 수학적 대칭성(Symmetry) 점수: **{similarity:.1f}%**")
    
    # AI 자동 분석 레포트 로직
    if similarity >= 85.0:
        st.success(f"💡 **[AI 자동 진단 결과: 우수]**\n두 샘플은 **{similarity:.1f}%**의 매우 높은 수학적 대칭성을 보입니다. 농도 오차 보정 후 완벽한 거울상(카이랄성)이 입증되었으며, 공정 조건(어닐링 시간 및 용매 증발)이 잘 통제되었습니다.")
    elif similarity >= 65.0:
        st.warning(f"⚠️ **[AI 자동 진단 결과: 보통]**\n대칭성이 **{similarity:.1f}%**로 보통 수준입니다. 미세한 구조적 흐트러짐이 관찰되며, 용매의 증발 속도 불균형이나 샘플 조제 과정에서의 미세 불순물 여부를 점검하세요.")
    else:
        st.error(f"🚨 **[AI 자동 진단 결과: 경고/불량]**\n대칭성이 **{similarity:.1f}%**로 매우 낮습니다! (Open/Close 조건 의심)\n어닐링(Annealing) 시간이 부족하여 고분자 자가조립이 불완전하거나, 외부 환경 영향으로 분자 배향이 왜곡되었을 가능성이 큽니다.")

    st.write("---")
    
    # [결과 2] Before & After 시각화 그래프
    plot_col1, plot_col2 = st.columns(2)
    
    with plot_col1:
        st.markdown("### ❌ Before: 원본 데이터 (농도 오차 발생)")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(df_r[0], df_r[2], label='R-form (Raw)', color='blue', linewidth=2)
        ax1.plot(df_s[0], df_s[2], label='S-form (Raw)', color='orange', linewidth=2)
        ax1.axhline(0, color='black', linewidth=0.8)
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('CD (mdeg)')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig1)

    with plot_col2:
        st.markdown("### ✨ After: 정규화 데이터 (스케일 보정 완료)")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(df_r[0], r_norm, label='R-form (Norm)', color='blue', linewidth=2)
        ax2.plot(df_s[0], s_norm, label='S-form (Norm)', color='orange', linestyle='--', linewidth=2)
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Normalized CD')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig2)

    # [결과 3] 핵심 피크 파장 & CD값 자동 추출 표
    st.write("---")
    st.markdown("### 🔍 핵심 피크(Peak) 수치 자동 추출")
    
    r_max_idx, r_min_idx = df_r[2].idxmax(), df_r[2].idxmin()
    s_max_idx, s_min_idx = df_s[2].idxmax(), df_s[2].idxmin()
    
    peak_data = {
        "구분": ["최대 피크 (Positive Peak)", "최소 피크 (Negative Peak)"],
        "R-form 파장 (nm)": [df_r[0].loc[r_max_idx], df_r[0].loc[r_min_idx]],
        "R-form CD값": [round(df_r[2].loc[r_max_idx], 2), round(df_r[2].loc[r_min_idx], 2)],
        "S-form 파장 (nm)": [df_s[0].loc[s_max_idx], df_s[0].loc[s_min_idx]],
        "S-form CD값": [round(df_s[2].loc[s_max_idx], 2), round(df_s[2].loc[s_min_idx], 2)]
    }
    
    peak_df = pd.DataFrame(peak_data)
    st.dataframe(peak_df, use_container_width=True, hide_index=True)
