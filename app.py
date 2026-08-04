import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import openai

# 웹사이트 기본 설정
st.set_page_config(page_title="Advanced Chiral Research Analyzer", layout="wide")

st.title("🧪 차세대 카이랄 분광 데이터 연구 및 공정 정밀 분석 플랫폼")
st.write("밀폐(Close) 조건 및 어닐링 시간별 데이터 기반: Executive Summary, 원인 진단, 그리고 구체적인 Action Plan을 도출합니다.")

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
# 2. 데이터 업로드 및 스마트 파일명 파서
# ==========================================
uploaded_files = st.file_uploader("📂 R/S form 및 시간별 CSV 파일 업로드 (예: R_60min.csv, S_60min.csv)", type=['csv'], accept_multiple_files=True)

if uploaded_files:
    st.write("---")
    st.header("📊 시간별 스펙트럼 비교 및 핵심 Peak 트렌드")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    peak_summary = []
    file_metadata = []
    
    data_tab1, data_tab2 = st.tabs(["📈 오버레이 스펙트럼", "📋 시간별 핵심 Peak 요약"])
    
    for f in uploaded_files:
        try:
            clean_name = f.name.replace('.csv', '').replace('.CSV', '')
            parts = clean_name.split('_')
            
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
                file_name="chiral_research_peaks.csv",
                mime="text/csv",
            )
        else:
            st.info("설정된 민감도에서 감지된 핵심 피크가 없습니다. 사이드바의 민감도를 낮춰보세요.")

    # ==========================================
    # 3. OpenAI API 연동: 연구원 맞춤형 심층 분석 보고서
    # ==========================================
    st.write("---")
    st.header("🤖 AI 기반 연구용 심층 분석 및 액션 플랜 리포트")
    st.write("제공해주신 연구원 전문 분석 포맷(Executive Summary, 원인 진단, Action Plan)에 맞춰 AI가 최종 결론을 도출합니다.")
    
    if st.button("🚀 AI 전문 연구 리포트 생성"):
        if not api_key:
            st.error("⚠️ 좌측 사이드바에 OpenAI API Key를 입력해 주세요!")
        elif not peak_summary:
            st.warning("⚠️ 분석할 피크 데이터가 없습니다.")
        else:
            with st.spinner("AI가 고도화된 연구 보고서(Executive Summary & Action Plan)를 작성하고 있습니다..."):
                try:
                    client = openai.OpenAI(api_key=api_key)
                    
                    prompt_data = peak_df.to_string(index=False)
                    meta_info = "\n".join(file_metadata)
                    
                    system_prompt = (
                        "당신은 고분자 화학 및 분광학 분야의 수석 선임 연구원입니다. "
                        "사용자는 밀폐(Close) 조건에서 어닐링 시간을 다변화하며(예: 40분, 60분, 90분, 120분 등) R-form과 S-form의 CD 스펙트럼 변화를 분석하고 있습니다. "
                        "출력 결과는 반드시 아래의 **지정된 포맷과 톤앤매너**를 정확히 준수하여 전문적이고 날카롭게 작성해 주세요:\n\n"
                        "### Executive Summary\n"
                        "- 실험 결과를 종합하여, 특정 시간 구간(예: 90~120분 등)에서의 신호 안정성, R/S형 각각의 peak intensity 트렌드, 그리고 mirror symmetry 충족 여부를 명확히 요약하세요.\n\n"
                        "### Overall Recommendation\n"
                        "- 현재 데이터에 기반하여 가장 적합한 최적의 Annealing 시간(예: 90분 등)을 선정하고 그 이유를 서술하세요.\n"
                        "- 만약 특정 시간대에서 **Peak가 갑자기 이동(Shift)**했다면, 가능한 원인으로 다음 중 타당한 것을 지목하세요: (Sample alignment, Instrument noise, Film thickness variation, Annealing temperature deviation).\n"
                        "- 특정 형태(예: S-form)에서 Positive/Negative peak가 유지되거나 변화하는 것이 분자배향 안정화, Molecular packing, Increased crystallinity, Reduced conformational disorder와 어떤 연관이 있는지 해석하세요.\n"
                        "- 현재 데이터만으로는 인접 시간대(예: 90분과 120분) 간의 차이가 크지 않다는 점을 진단하세요.\n\n"
                        "### 추천 Action Plan (체크박스 형태로 명시)\n"
                        "추가 실험이나 검토가 필요한 항목을 아래 형식의 체크박스로 정확히 제안하세요:\n"
                        "□ [추가 시간 측정 제안, 예: 100분 추가 측정]\n"
                        "□ [추가 시간 측정 제안, 예: 110분 추가 측정]\n"
                        "□ Temperature variation 검토\n"
                        "□ Repeat measurement (재현성 검증)"
                    )
                    
                    user_prompt = f"### 실험 파일 정보\n{meta_info}\n\n### 시간별 상위 핵심 피크 데이터\n{prompt_data}\n\n위 데이터를 바탕으로 제시된 포맷(Executive Summary, Overall Recommendation, 구체적 원인 진단 및 체크박스 추천 Action Plan)에 정확히 맞춘 최고급 연구 분석 보고서를 작성해 줘."
                    
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.2
                    )
                    
                    st.success("✅ AI 연구 분석 리포트 생성 완료!")
                    st.markdown(f"### 📑 최종 연구 분석 보고서")
                    st.write(response.choices[0].message.content)
                    
                except Exception as e:
                    st.error(f"OpenAI API 호출 중 오류가 발생했습니다: {e}")
