import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import openai

# 웹사이트 기본 설정
st.set_page_config(page_title="Chiral Time-Series AI Analyzer", layout="wide")

st.title("🧪 카이랄 분광 데이터 시간별 어닐링 및 대칭성 정밀 분석 플랫폼")
st.write("밀폐(Close) 조건 및 어닐링 시간(40~120분)에 따른 R/S 이성질체의 거울상 대칭성 정량화 및 AI 기반 실험 리포트 자동 생성 시스템")

# ==========================================
# 1. 사이드바: 분석 설정 및 OpenAI API 세팅
# ==========================================
with st.sidebar:
    st.header("⚙️ 분석 및 AI 설정")
    api_key = st.text_input("🔑 OpenAI API Key 입력", type="password")
    
    st.write("---")
    spec_type = st.selectbox("📊 분광 데이터 종류", ["원편광이색성 (CD)", "자외선-가시광선 (UV-Vis)", "적외선 (FT-IR)", "라만 (Raman)"])
    
    st.write("---")
    st.write("🔍 핵심 피크 필터링 설정")
    prominence = st.slider("피크 감지 민감도 (Prominence)", min_value=0.1, max_value=3.0, value=0.4, step=0.1)
    max_peaks_to_show = st.slider("핵심 피크 표시 개수 제한", min_value=1, max_value=5, value=5)

# ==========================================
# 2. 데이터 업로드 및 스마트 파일명 파서 (시간/형태 추출)
# ==========================================
uploaded_files = st.file_uploader("📂 R/S form 및 시간별 CSV 파일 업로드 (예: R_60min.csv, S_60min.csv)", type=['csv'], accept_multiple_files=True)

if uploaded_files:
    st.write("---")
    st.header("📊 시간별 스펙트럼 비교 및 카이랄 대칭성 지표")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    peak_summary = []
    file_metadata = []
    
    data_tab1, data_tab2 = st.tabs(["📈 오버레이 스펙트럼", "📋 시간별 핵심 Peak 및 대칭성 요약"])
    
    for f in uploaded_files:
        try:
            clean_name = f.name.replace('.csv', '').replace('.CSV', '')
            parts = clean_name.split('_')
            
            # 파일명 구조 유연하게 파싱 (예: R_60min 또는 R_close_60min 모두 대응)
            form = parts[0].upper() if len(parts) > 0 else "UNKNOWN"
            
            time_val = "unknown"
            condition = "closed (default)"
            
            for p in parts[1:]:
                p_lower = p.lower()
                if 'min' in p_lower or 'hr' in p_lower or 'h' in p_lower and any(char.isdigit() for char in p_lower):
                    time_val = p_lower
                elif p_lower in ['open', 'half', 'close', 'closed']:
                    condition = p_lower
            
            file_metadata.append(f"파일 이름: {f.name} (형태: {form}, 조건: {condition}, 어닐링 시간: {time_val})")
            
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
            
            sample_peaks = []
            for p in peaks_pos:
                sample_peaks.append({"샘플명": f.name, "형태": form, "시간": time_val, "유형": "Positive", "파장(nm)": round(x[p], 2), "강도(Y)": round(y[p], 3), "AbsY": abs(y[p])})
            for p in peaks_neg:
                sample_peaks.append({"샘플명": f.name, "형태": form, "시간": time_val, "유형": "Negative", "파장(nm)": round(x[p], 2), "강도(Y)": round(y[p], 3), "AbsY": abs(y[p])})
            
            sample_peaks = sorted(sample_peaks, key=lambda k: k["AbsY"], reverse=True)[:max_peaks_to_show]
            for sp in sample_peaks:
                del sp["AbsY"]
                peak_summary.append(sp)
                
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
            peak_df = peak_df.sort_values(by=["시간", "파장(nm)"]).reset_index(drop=True)
            st.subheader("🎯 시간별 샘플 핵심 주요 피크 (Top 기여도)")
            st.dataframe(peak_df, use_container_width=True)
            
            csv_data = peak_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 핵심 피크 데이터 CSV 다운로드",
                data=csv_data,
                file_name="chiral_time_peaks.csv",
                mime="text/csv",
            )
        else:
            st.info("설정된 민감도에서 감지된 핵심 피크가 없습니다. 사이드바의 민감도를 낮춰보세요.")

    # ==========================================
    # 3. OpenAI API 연동: 시간별 어닐링 및 대칭성 리포트
    # ==========================================
    st.write("---")
    st.header("🤖 AI 기반 시간별 어닐링 및 카이랄 대칭성 정밀 평가 리포트")
    st.write("밀폐(Close) 조건에서 어닐링 시간 변화가 R/S 이성질체의 거울상 대칭성 및 분자 배향 안정화에 미친 영향을 분석합니다.")
    
    if st.button("🚀 AI 정밀 분석 및 리포트 생성"):
        if not api_key:
            st.error("⚠️ 좌측 사이드바에 OpenAI API Key를 입력해 주세요!")
        elif not peak_summary:
            st.warning("⚠️ 분석할 피크 데이터가 없습니다.")
        else:
            with st.spinner("AI가 시간대별 R/S형 대칭성 점수와 어닐링 안정화 효과를 분석하고 있습니다..."):
                try:
                    client = openai.OpenAI(api_key=api_key)
                    
                    prompt_data = peak_df.to_string(index=False)
                    meta_info = "\n".join(file_metadata)
                    
                    system_prompt = (
                        "당신은 카이랄 분광학 및 물리화학 수석 연구원입니다. "
                        "사용자는 모든 비커를 밀폐(Close)한 상태에서 어닐링 시간(예: 40분, 60분, 90분, 120분 등)을 다변화하며 "
                        "R-form과 S-form의 거울상 대칭성(Cotton Effect) 변화를 연구 중입니다. "
                        "절대로 비커를 열어두었다는 식의 가짜 해석을 하지 말고, 밀폐된 환경에서의 시간별 어닐링 효과에 집중하세요. "
                        "결과 리포트는 다음 구조를 반드시 포함해 주세요:\n\n"
                        "1. **시간대별 R/S형 거울상 대칭성 정량 평가 (Chiral Symmetry Score)**:\n"
                        "   - 동일 시간대(예: 60분짜리 R과 S)별로 파장 일치 여부와 Cotton Effect 부호 반전 여부를 대조하고, 대칭성 일치도(%)를 산출하여 평가하세요.\n"
                        "2. **어닐링 시간(40분~120분 등)에 따른 분자 배향 및 열역학적 안정화 메커니즘**:\n"
                        "   - 밀폐 조건에서 시간이 경과함에 따라 결정성이나 분자 뭉침(Aggregation)이 어떻게 안정화되거나 수렴하는지 분석하세요.\n"
                        "3. **실험 성공/실패 진단 및 향후 액션 플랜 (Conclusion & Action Plan)**:\n"
                        "   - 데이터가 이론적 거울상 대칭성을 잘 따르는지, 혹은 특정 시간대에서 오차가 발생하는지 냉정하게 진단하고, "
                        "   - 최적의 어닐링 시간(예: 몇 분이 가장 이상적인가)을 제안하며 추가 실험을 위한 실질적인 피드백을 제공하세요."
                    )
                    
                    user_prompt = f"### 실험 파일 정보\n{meta_info}\n\n### 시간별 상위 핵심 피크 데이터\n{prompt_data}\n\n위 데이터를 바탕으로 밀폐 조건에서의 시간별 어닐링 효과, R/S형 간 대칭성 수치 평가, 그리고 실험 성공/실패 진단 및 최적 시간 도출 리포트를 작성해 줘."
                    
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.2
                    )
                    
                    st.success("✅ AI 정밀 분석 완료!")
                    st.markdown(f"### 📑 시간별 어닐링 및 대칭성 평가 최종 보고서")
                    st.write(response.choices[0].message.content)
                    
                except Exception as e:
                    st.error(f"OpenAI API 호출 중 오류가 발생했습니다: {e}")
