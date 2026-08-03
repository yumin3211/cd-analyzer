import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 웹사이트 제목과 설명
st.set_page_config(page_title="Chirality Analyzer", layout="wide")
st.title("📊 CD 스펙트럼 대칭성 자동 분석기")
st.write("농도 오차(진폭)를 자동으로 정규화(Normalization)하고, 수학적 대칭성 점수를 도출합니다.")

# 파일 업로드 칸 만들기
col1, col2 = st.columns(2)
with col1:
    file_r = st.file_uploader("R-form CSV 파일을 올려주세요", type=['csv'])
with col2:
    file_s = st.file_uploader("S-form CSV 파일을 올려주세요", type=['csv'])

# 파일 2개가 모두 올라오면 분석 시작
if file_r and file_s:
    # 1. 데이터 읽기 및 텍스트 지우기
    df_r = pd.read_csv(file_r, skiprows=21, header=None)
    df_s = pd.read_csv(file_s, skiprows=21, header=None)
    
    df_r[0] = pd.to_numeric(df_r[0], errors='coerce')
    df_r[2] = pd.to_numeric(df_r[2], errors='coerce')
    df_r = df_r.dropna(subset=[0, 2])
    
    df_s[0] = pd.to_numeric(df_s[0], errors='coerce')
    df_s[2] = pd.to_numeric(df_s[2], errors='coerce')
    df_s = df_s.dropna(subset=[0, 2])
    
    # 2. 정규화 (Normalization)
    r_norm = df_r[2] / df_r[2].abs().max()
    s_norm = df_s[2] / df_s[2].abs().max()
    
    # 3. 대칭성 점수 계산
    try:
        similarity = np.corrcoef(r_norm, -s_norm)[0, 1] * 100
    except:
        similarity = 0
    
    st.divider() 
    
    # 결과 점수 크게 출력
    st.subheader(f"✅ 수학적 대칭성(Symmetry) 점수: **{similarity:.1f}%**")
    st.write("---")
    
    # 🚀 [업그레이드 1: Before & After 시각화]
    # 화면을 좌우 두 칸으로 나누어 비교합니다.
    plot_col1, plot_col2 = st.columns(2)
    
    with plot_col1:
        st.markdown("### ❌ Before: 원본 데이터 (농도 오차 발생)")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        # 원본 데이터는 S-form을 뒤집지 않고 날것 그대로 보여줍니다.
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
        # 보정된 데이터는 S-form을 뒤집어서(-s_norm) 완벽한 거울상인지 확인합니다.
        ax2.plot(df_r[0], r_norm, label='R-form (Norm)', color='blue', linewidth=2)
        ax2.plot(df_s[0], -s_norm, label='S-form (Flipped, Norm)', color='orange', linestyle='--', linewidth=2)
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Normalized CD')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig2)
