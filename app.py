import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import openai

# 웹사이트 기본 설정
st.set_page_config(page_title="Advanced Chiral AI Analyzer", layout="wide")

st.title("🧪 차세대 카이랄 분광 데이터 정밀 분석 및 공정 평가 플랫폼")
st.write("비커 개폐 조건(Open/Half/Close)에 따른 R/S 이성질체의 거울상 대칭성 정량화 및 AI 기반 실험 리포트 자동 생성 시스템")

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
    # 민감도를 높여서 정말 중요한 상위 피크만 잡도록 기본값 조절
    prominence = st.slider("피크 감지 민감도 (Prominence)", min_value=0.1, max_value=3.0, value=0.5, step=0.1)
    max_peaks_to_show = st.slider("핵심 피크 표시 개수 제한", min_value=1, max_value=5, value=5)

# ==========================================
# 2. 데이터 업로드 및 스마트 파싱
# ==========================================
uploaded_files = st.file_uploader("📂 R/S form 및 조건별 CSV 파일 업로드 (다중 선택 가능)", type=['csv'], accept_multiple_files=True)

if uploaded_files:
    st.write("---")
    st.header("📊 다중 샘플 스펙트럼 비교 및 카이랄 대칭성 지표")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    peak_summary = []
    file_metadata = []
    
    data_tab1, data_tab2 = st.tabs(["📈 오버레이 스펙트럼", "📋 핵심 Peak 및 카이랄 요약"])
    
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
            
            # 강도(Y값 절대값)가 큰 상위 피크만 추출하기 위해 임시 리스트에 담기
            sample_peaks = []
            for p in peaks_pos:
                sample_peaks.append({"샘플명": f.name, "형태(Form)": form, "비커조건": condition, "유형": "Positive", "파장(nm)": round(x[p], 2), "강도(Y)": round(y[p], 3), "AbsY": abs(y[p])})
            for p in peaks_neg:
                sample_peaks.append({"샘플명": f.name, "형태(Form)": form, "비커조건": condition, "유형": "Negative", "파장(nm)": round(x[p], 2), "강도(Y)": round(y[p], 3), "AbsY": abs(y[p])})
            
            # 강도가 큰 순서대로 정렬하여 상위 N개만 채택 (의미 없는 잡다한 피크 배제)
            sample_peaks = sorted(sample_peaks, key=lambda k: k["AbsY"], reverse=True)[:max_peaks_to_show]
            for sp in sample_peaks:
                del sp["AbsY"] # 정렬용 임시 키 삭제
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
            peak_df = peak_df.sort_values(by=["파장(nm)"]).reset_index(drop=True)
            st.subheader("🎯 샘플별 핵심 주요 피크 (Top 기여도)")
            st.dataframe(peak_df, use_container_width=True)
            
            # 엑셀 다운로드 버튼 추가
            csv_data = peak_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 핵심 피크 데이터 CSV 다운로드",
                data=csv_data,
                file_name="chiral_key_peaks.csv",
                mime="text/csv",
            )
        else:
            st.info("설정된 민감도에서 감지된 핵심 피크가 없습니다. 사이드바의 민감도를 낮춰보세요.")

    # ==========================================
    # 3. OpenAI API 연동: 정량적 대칭성 및 결론/대책 리포트
    # ==========================================
    st.write("---")
    st.header("🤖 AI 기반 카이랄 대칭성 정량 분석 및 실험 평가 리포트")
    st.write("R/S 이성질체 간의 대칭성 수치화, 이론 부합도 평가, 그리고 구체적인 후속 개선 대책을 도출합니다.")
    
    if st.button("🚀 AI 정밀 분석 및 리포트 생성"):
        if not api_key:
            st.error("⚠️ 좌측 사이드바에 OpenAI API Key를 입력해 주세요!")
        elif not peak_summary:
            st.warning("⚠️ 분석할 피크 데이터가 없습니다.")
        else:
            with st.spinner("AI가 거울상 대칭성 수치를 계산하고 실험 성공 여부 및 개선 대책을 수립하고 있습니다..."):
                try:
                    client = openai.OpenAI(api_key=api_key)
                    
                    prompt_data = peak_df.to_string(index=False)
                    meta_info = "\n".join(file_metadata)
                    
                    system_prompt = (
                        "당신은 카이랄 분광학 및 물리화학 수석 연구원입니다. "
                        "사용자는 비커 개폐 조건(Open/Half/Close)에 따른 R-form과 S-form의 카이랄성(거울상 대칭성)을 연구 중입니다. "
                        "결과 리포트는 반드시 다음 구조와 내용을 포함하여 전문적이고 날카롭게 작성해 주세요:\n\n"
                        "1. **카이랄 거울상 대칭성 정량 평가 (Chiral Symmetry Score)**:\n"
                        "   - R형과 S형의 주요 피크 파장 및 Cotton Effect 부호 반전 여부를 대조하고, 대칭성이 얼마나 유지되었는지 백분율(%) 형태의 대칭성 지표를 추정/산출하여 평가하세요.\n"
                        "2. **비커 개폐 조건에 따른 용매 증발 제어 메커니즘 분석**:\n"
                        "   - Open/Half/Close 조건에 따른 증발 속도 차이가 분자 배향과 카이랄 특성 발현에 미친 영향을 열역학적으로 분석하세요.\n"
                        "3. **실험 결과 종합 평가 및 향후 대책 (Conclusion & Action Plan)**:\n"
                        "   - 이번 실험 결과가 이론적인 거울상 대칭성 모델에 잘 부합하는지, 아니면 데이터가 분산되거나 불안정하여 오차가 발생했는지 냉정하게 진단하세요.\n"
                        "   - 만약 오차가 있다면 '재실험 시 밀폐 조건을 강화할 것', '온도 구배를 일정하게 유지할 것' 등 앞으로 어떻게 실험을 보완해야 할지 구체적이고 실질적인 액션 플랜을 제시하세요."
                    )
                    
                    user_prompt = f"### 실험 파일 정보\n{meta_info}\n\n### 상위 핵심 피크 데이터\n{prompt_data}\n\n위 데이터를 바탕으로 대칭성 수치 평가, 증발 메커니즘, 그리고 실험의 성공/실패 여부 진단 및 향후 개선 대책을 포함한 종합 리포트를 작성해 줘."
                    
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.2
                    )
                    
                    st.success("✅ AI 정밀 분석 완료!")
                    st.markdown(f"### 📑 카이랄 대칭성 및 공정 평가 최종 보고서")
                    st.write(response.choices[0].message.content)
                    
                except Exception as e:
                    st.error(f"OpenAI API 호출 중 오류가 발생했습니다: {e}")
