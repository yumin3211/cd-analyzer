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
    df_r = pd.read_csv(file_r, skiprows=21, header=None)
    df_s = pd.read_csv(file_s, skiprows=21, header=None)
    
    df_r[0] = pd.to_numeric(df_r[0], errors='coerce')
    df_r[2] = pd.to_numeric(df_r[2], errors='coerce')
    df_r = df_r.dropna(subset=[0, 2])
    
    df_s[0] = pd.to_numeric(df_s[0], errors='coerce')
    df_s[2] = pd.to_numeric(df_s[2], errors='coerce')
    df_s = df_s.dropna(subset=[0, 2])
    
    r_norm = df_r[2] / df_r[2].abs().max()
    s_norm = df_s[2] / df_s[2].abs().max()
    
    try:
        similarity = np.corrcoef(r_norm, -s_norm)[0, 1] * 100
    except:
        similarity = 0
    
    st.divider() 
    
    st.subheader(f"✅ 대칭성(Symmetry) 점수: **{similarity:.1f}%**")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_r[0], r_norm, label='R-form', color='blue', linewidth=2)
    ax.plot(df_s[0], -s_norm, label='S-form (Flipped)', color='orange', linestyle='--', linewidth=2)
    
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Normalized CD')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    st.pyplot(fig)
