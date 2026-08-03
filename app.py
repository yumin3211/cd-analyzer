import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import openai

# 웹사이트 기본 설정
st.set_page_config(page_title="AI Spectral Analyzer", layout="wide")

st.title("🔬 AI 기반 범용 분광 데이터 자동 분석 플랫폼")
st.write("다중 스펙트럼(CD, UV-Vis, IR 등) 데이터를 비교하고, OpenAI API를 활용하여 실험 결과를 자동으로 해석합니다.")

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
    # scipy의 find_peaks 민감도 조절
    prominence = st.slider("피크 감지 민감도 (Prominence)", min_value=0.01, max_value=2.0, value=0.1, step=0.05)
    
# ==========================================
# 2. 데이터 업로드 및 파싱 (범용)
# ==========================================
uploaded_files = st.file_uploader("📂 분광 데이터 CSV 파일 업로드 (다중 선택 가능)", type=['csv'], accept_multiple_files=True)

if uploaded_files:
    st.write("---")
    st.header("📊 다중 샘플 스펙트럼 비교 및 Peak 자동 추출")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    peak_summary = []
    
    # 탭을 활용하여 원본 데이터와 피크 데이터를 나눠서 보여줌
    data_tab1, data_tab2 = st.tabs(["📈 오버레이 스펙트럼", "📋 추출된 Peak 데이터"])
    
    for f in uploaded_files:
        try:
            # [범용 파싱 로직] 파일의 형태와 상관없이 숫자 데이터만 추출
            df = pd.read_csv(f)
            
            # 1열(X축-파장)과 마지막 열(Y축-강도/흡광도)을 강제로 숫자로 변환 (텍스트 헤더 무시)
            x_col = df.columns[0]
            y_col = df.columns[-1]
            
            df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
            df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
            df = df.dropna(subset=[x_col, y_col]) # NaN 값 제거
            
            x = df[x_col].values
            y = df[y_col].values
            
            # Scipy 알고리즘을 이용한 피크 자동 탐지
            peaks_pos, _ = find_peaks(y, prominence=prominence) # 양(+)의 피크
            peaks_neg, _ = find_peaks(-y, prominence=prominence) # 음(-)의 피크 (CD 스펙트럼용)
            
            # 그래프 그리기
            ax.plot(x, y, label=f.name, linewidth=1.5)
            ax.plot(x[peaks_pos], y[peaks_pos], "x", color='red', markersize=6) # 피크 표시
            ax.plot(x[peaks_neg], y[peaks_neg], "x", color='blue', markersize=6)
            
            # 피크 데이터 저장
            for p in peaks_pos:
                peak_summary.append({"샘플명": f.name, "구분": "Positive Peak", "위치(X)": round(x[p], 2), "강도(Y)": round(y[p], 3)})
            for p in peaks_neg:
                peak_summary.append({"샘플명": f.name, "구분": "Negative Peak", "위치(X)": round(x[p], 2), "강도(Y)": round(y[p], 3)})
                
        except Exception as e:
            st.error(f"{f.name} 처리 중 오류 발생: {e}")

    # 그래프 세팅 마무리
    ax.axhline(0, color='black', linewidth=0.8)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    with data_tab1:
        st.pyplot(fig)
        
    with data_tab2:
        if peak_summary:
            peak_df = pd.DataFrame(peak_summary)
            # 위치(X)를 기준으로 오름차순 정렬
            peak_df = peak_df.sort_values(by=["위치(X)"]).reset_index(drop=True)
            st.dataframe(peak_df, use_container_width=True)
        else:
            st.info("설정된 민감도에서 감지된 피크가 없습니다. 왼쪽 사이드바에서 민감도를 조절해 보세요.")

    # ==========================================
    # 3. OpenAI API 연동 AI 자동 리포트 생성
    # ==========================================
    st.write("---")
    st.header("🤖 LLM 기반 실험 결과 자동 해석 리포트")
    st.write("추출된 다중 샘플의 Peak 데이터를 바탕으로 AI가 화학/물리적 의미를 해석합니다.")
    
    if st.button("🚀 AI 결과 분석 실행"):
        if not api_key:
            st.error("⚠️ 좌측 사이드바에 OpenAI API Key를 입력해 주세요!")
        elif not peak_summary:
            st.warning("⚠️ 추출된 피크 데이터가 없어 분석할 수 없습니다.")
        else:
            with st.spinner("AI가 데이터를 분석하고 리포트를 작성하고 있습니다..."):
                try:
                    # OpenAI 클라이언트 생성 (최신 API 방식)
                    client = openai.OpenAI(api_key=api_key)
                    
                    # LLM에게 던질 프롬프트(명령어) 작성
                    prompt_data = peak_df.to_string(index=False)
                    system_prompt = "당신은 반도체 및 신소재 데이터 분석을 전문으로 하는 수석 데이터 사이언티스트입니다. 제공된 분광 데이터의 피크 수치를 바탕으로 샘플 간의 차이, 물질의 구조적 특성, 그리고 데이터가 의미하는 물리/화학적 인사이트를 전문적인 보고서 형식으로 작성해 주세요."
                    user_prompt = f"다음은 {spec_type} 실험을 통해 얻은 다중 샘플의 피크(Peak) 데이터입니다.\n\n{prompt_data}\n\n이 데이터를 비교 분석하여 종합적인 결과 해석 리포트를 작성해 줘."
                    
                    # API 호출
                    response = client.chat.completions.create(
                        model="gpt-4o-mini", # 비용 효율적인 최신 모델
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.3 # 분석적인 답변을 위해 창의성 낮춤
                    )
                    
                    # 결과 출력
                    st.success("✅ AI 분석 완료!")
                    st.markdown(f"### 📑 {spec_type} 종합 분석 리포트")
                    st.write(response.choices[0].message.content)
                    
                except Exception as e:
                    st.error(f"OpenAI API 호출 중 오류가 발생했습니다: {e}")
