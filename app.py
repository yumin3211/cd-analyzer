import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import openai

# 웹사이트 기본 설정
st.set_page_config(page_title="Chiral Spectroscopic AI Analyzer", layout="wide")

st.title("🧪 AI 기반 카이랄 분광 데이터 및 공정 조건 분석 플랫폼")
st.write("비커 개폐 조건(Open/Half/Close) 및 어닐링 시간에 따른 거울상 이성질체(R/S) 대칭성과 카이랄 특성을 정밀 분석합니다.")

# ==========================================
# 1. 사이드바: 분석 설정 및 OpenAI API 세팅
# ==========================================
with st.sidebar:
    st.header("⚙️ 분석 및 AI 설정")
    st.info("AI 리포트 생성을 위해 OpenAI API Key가 필요합니다.")
    api_key = st.text_input("🔑 OpenAI API Key 입력", type="password")
    
    st.write("---")
    spec_type = st.selectbox("📊 분광 데이터 종류", ["원편광이색성 (CD)", "자외선-가시광선 (UV-Vis)", "적외선 (FT-IR)", "라만 (Raman)", "기타"])
    
    st.write("---")
    st.write("🔍 피크(Peak) 추출 알고리즘 설정")
    prominence = st.slider("피크 감지 민감도 (Prominence)", min_value=0.01, max_value=2.0, value=0.1, step=0.05)
    
# ==========================================
# 2. 데이터 업로드 및 스마트 파싱 (조건 자동 분류)
# ==========================================
uploaded_files = st.file_uploader("📂 R/S form 및 조건별 CSV 파일 업로드 (다중 선택 가능)", type=['csv'], accept_multiple_files=True)

if uploaded_files:
    st.write("---")
    st.header("📊 다중 샘플 스펙트럼 비교 및 카이랄 대칭성 분석")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    peak_summary = []
    file_metadata = []
    
    data_tab1, data_tab2 = st.tabs(["📈 오버레이 스펙트럼", "📋 카이랄/Peak 데이터 요약"])
    
    for f in uploaded_files:
        try:
            clean_name = f.name.replace('.csv', '').replace('.CSV', '')
            parts = clean_name.split('_')
            form = parts[0].upper() if len(parts) > 0 else "UNKNOWN"
            condition = parts[1].lower() if len(parts) > 1 else "unknown"
            time_val = parts[2] if len(parts) > 2 else "unknown"
            
            file_metadata.append(f"파일 이름: {f.name} (형태: {form}, 조건: {condition}, 시간: {time_val})")
            
            f.seek(0) 
            try:
                df = pd.read_csv(f)
                x_col = df.columns[0]
                y_col = df.columns[-1]
            except:
                f.seek(0)
                df = pd.read_csv(f, skiprows=21, header=None)
                x_col = 0
                y_col = 2
            
            df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
            df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
            df = df.dropna(subset=[x_col, y_col]) 
            
            x = df[x_col].values
            y = df[y_col].values
            
            peaks_pos, _ = find_peaks(y, prominence=prominence)
            peaks_neg, _ = find_peaks(-y, prominence=prominence)
            
            ax.plot(x, y, label=f.name, linewidth=1.5)
            ax.plot(x[peaks_pos], y[peaks_pos], "x", color='red', markersize=6)
            ax.plot(x[peaks_neg], y[peaks_neg], "x", color='blue', markersize=6)
            
            for p in peaks_pos:
                peak_summary.append({"샘플명": f.name, "형태(Form)": form, "비커조건": condition, "구분": "Positive Peak", "위치(X/nm)": round(x[p], 2), "강도(Y)": round(y[p], 3)})
            for p in peaks_neg:
                peak_summary.append({"샘플명": f.name, "형태(Form)": form, "비커조건": condition, "구분": "Negative Peak", "위치(X/nm)": round(x[p], 2), "강도(Y)": round(y[p], 3)})
                
        except Exception as e:
            st.error(f"{f.name} 처리 중 오류 발생: {e}")

    ax.axhline(0, color='black', linewidth=0.8)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    with data_tab1:
        st.pyplot(fig)
        
    with data_tab2:
        if peak_summary:
            peak_df = pd.DataFrame(peak_summary)
            peak_df = peak_df.sort_values(by=["위치(X/nm)"]).reset_index(drop=True)
            st.dataframe(peak_df, use_container_width=True)
        else:
            st.info("설정된 민감도에서 감지된 피크가 없습니다. 왼쪽 사이드바에서 민감도를 조절해 보세요.")

    # ==========================================
    # 3. OpenAI API 연동: 카이랄 특성 및 공정 조건 심층 해석
    # ==========================================
    st.write("---")
    st.header("🤖 LLM 기반 카이랄성 및 증발 제어 공정 심층 해석 리포트")
    st.write("비커 개폐 조건(Open/Half/Close) 또는 시간에 따른 용매 증발 제어가 R/S 이성질체의 거울상 대칭성(Cotton Effect)에 미친 영향을 분석합니다.")
    
    if st.button("🚀 AI 카이랄 공정 분석 실행"):
        if not api_key:
            st.error("⚠️ 좌측 사이드바에 OpenAI API Key를 입력해 주세요!")
        elif not peak_summary:
            st.warning("⚠️ 분석할 피크 데이터가 없습니다.")
        else:
            with st.spinner("AI가 거울상 대칭성과 공정 조건별 메커니즘을 분석하고 있습니다..."):
                try:
                    client = openai.OpenAI(api_key=api_key)
                    
                    prompt_data = peak_df.to_string(index=False)
                    meta_info = "\n".join(file_metadata)
                    
                    system_prompt = (
                        "당신은 고분자 화학 및 분광학(CD 스펙트럼) 분야의 수석 연구원입니다. "
                        "사용자는 비커 조건(Open/Half/Close)이나 어닐링 시간을 조절하며 "
                        "R-form과 S-form 물질의 카이랄성(거울상 대칭성, Cotton Effect) 변화를 연구하고 있습니다. "
                        "제공된 파일 메타데이터와 피크 데이터를 바탕으로 다음 내용을 포함한 전문적인 연구 분석 리포트를 작성해 주세요:\n"
                        "1. R-form과 S-form 간의 거울상 대칭성(Cotton Effect 및 부호 반전 여부) 평가\n"
                        "2. 조건(개폐 여부 또는 시간 등)에 따른 용매 증발 속도 제어가 분자 배향과 카이랄성 발현에 미친 열역학적/공정적 영향 해석\n"
                        "3. 데이터에 기반한 최적의 공정 조건 도출"
                    )
                    
                    user_prompt = f"### 실험 파일 정보\n{meta_info}\n\n### 추출된 피크 데이터\n{prompt_data}\n\n위 데이터를 바탕으로 카이랄성 변화와 공정 특성을 심층 분석해 줘."
                    
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.3
                    )
                    
                    st.success("✅ AI 카이랄 공정 분석 완료!")
                    st.markdown(f"### 📑 카이랄 분광 분석 및 공정 리포트")
                    st.write(response.choices[0].message.content)
                    
                except Exception as e:
                    st.error(f"OpenAI API 호출 중 오류가 발생했습니다: {e}")
