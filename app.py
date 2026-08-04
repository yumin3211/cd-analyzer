import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import openai
import io

# 웹사이트 기본 설정
st.set_page_config(page_title="Ultimate Chiral Spectroscopic Analyzer", layout="wide")

st.title("🔬 카이랄 분광 데이터 종합 분석 및 논문형 리포트 자동화 플랫폼")
st.write("시간/조건별 R·S 이성질체 정량 비교, 이상 탐지, 논문형 리포트 생성 및 분석 결과 아카이브(Save) 시스템입니다.")

# ==========================================
# 0. 세션 상태(Session State) 초기화
# ==========================================
if 'saved_reports' not in st.session_state:
    st.session_state.saved_reports = []

# ==========================================
# 1. 사이드바: 분석 설정 및 OpenAI API 세팅
# ==========================================
with st.sidebar:
    st.header("⚙️ 분석 및 AI 설정")
    api_key = st.text_input("🔑 OpenAI API Key 입력", type="password")
    
    st.write("---")
    spec_type = st.selectbox("📊 분광 데이터 종류", ["원편광이색성 (CD)", "자외선-가시광선 (UV-Vis)", "적외선 (FT-IR)", "라만 (Raman)"])
    
    st.write("---")
    st.write("🔍 피크(Peak) 및 이상 탐지 설정")
    prominence = st.slider("피크 감지 민감도 (Prominence)", min_value=0.1, max_value=3.0, value=0.5, step=0.1)
    max_peaks_to_show = st.slider("핵심 피크 표시 개수 제한", min_value=1, max_value=10, value=5)

# ==========================================
# 2. 데이터 업로드 및 스마트 파싱 / 대칭성·이동량 정량화
# ==========================================
uploaded_files = st.file_uploader("📂 R/S form 및 시간/조건별 CSV 파일 업로드", type=['csv'], accept_multiple_files=True)

if uploaded_files:
    st.write("---")
    st.header("📊 스펙트럼 오버레이 및 R/S 이성질체 간 거울상 대칭성 정량 분석")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    peak_summary = []
    file_metadata = []
    time_groups = {}
    anomaly_reports = []
    
    data_tab1, data_tab2, data_tab3, data_tab4 = st.tabs([
        "📈 오버레이 스펙트럼", 
        "📊 조건별 R vs S 거울상 대칭성 비교", 
        "📋 전체 핵심 Peak 요약",
        "📂 저장된 분석 기록 (Save Archive)"
    ])
    
    for f in uploaded_files:
        try:
            clean_name = f.name.replace('.csv', '').replace('.CSV', '')
            parts = clean_name.split('_')
            
            form = "UNKNOWN"
            if len(parts) > 0:
                if parts[0].upper() in ["R", "S"]:
                    form = parts[0].upper()
                elif "R" in parts[0].upper(): form = "R"
                elif "S" in parts[0].upper(): form = "S"

            time_val = "unknown"
            condition = "closed"
            
            for p in parts[1:]:
                p_lower = p.lower()
                if 'min' in p_lower or 'hr' in p_lower or ('h' in p_lower and any(char.isdigit() for char in p_lower)) or p_lower.isdigit():
                    time_val = p_lower
                elif p_lower in ['open', 'half', 'h', 'close', 'closed', 'c']:
                    condition = p_lower
            
            # R과 S를 올바르게 짝짓기 위해 시간과 조건만 그룹 키로 사용
            group_key = f"{time_val} ({condition})"
            file_metadata.append(f"파일: {f.name} (형태: {form}, 조건: {condition}, 시간: {time_val})")
            
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
            
            total_peaks_detected = len(peaks_pos) + len(peaks_neg)
            if total_peaks_detected > 20:
                anomaly_reports.append(f"⚠️ **{f.name}**: 비정상적으로 많은 피크({total_peaks_detected}개) 검출. Instrument noise 점검 필요.")
            if len(y) > 0 and max(abs(y)) < 0.5:
                anomaly_reports.append(f"⚠️ **{f.name}**: 신호 세기가 매우 낮음(Max < 0.5). 농도 또는 필름 두께 확인 필요.")
            
            ax.plot(x, y, label=f.name, linewidth=1.5)
            ax.plot(x[peaks_pos], y[peaks_pos], "x", color='red', markersize=6)
            ax.plot(x[peaks_neg], y[peaks_neg], "x", color='blue', markersize=6)
            
            sample_peaks = []
            for p in peaks_pos:
                sample_peaks.append({"샘플명": f.name, "형태": form, "조건그룹": group_key, "유형": "Positive", "파장(nm)": round(x[p], 2), "강도(Y)": round(y[p], 3), "AbsY": abs(y[p])})
            for p in peaks_neg:
                sample_peaks.append({"샘플명": f.name, "형태": form, "조건그룹": group_key, "유형": "Negative", "파장(nm)": round(x[p], 2), "강도(Y)": round(y[p], 3), "AbsY": abs(y[p])})
            
            if group_key not in time_groups:
                time_groups[group_key] = {"R": [], "S": []}
            if form == "R":
                time_groups[group_key]["R"].extend(sample_peaks)
            elif form == "S":
                time_groups[group_key]["S"].extend(sample_peaks)

            sample_peaks = sorted(sample_peaks, key=lambda k: k["AbsY"], reverse=True)[:max_peaks_to_show]
            for sp in sample_peaks:
                peak_summary.append(sp)
                
        except Exception as e:
            st.error(f"{f.name} 처리 중 오류 발생: {e}")

    ax.axhline(0, color='black', linewidth=0.8)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 동일 조건 그룹 내에서 R과 S의 거울상 대칭성(Mirror Symmetry) 정량 비교 계산
    symmetry_results = []
    for g_key, forms in time_groups.items():
        r_peaks = forms["R"]
        s_peaks = forms["S"]
        
        if r_peaks and s_peaks:
            r_top = sorted(r_peaks, key=lambda k: k["AbsY"], reverse=True)[0]
            s_top = sorted(s_peaks, key=lambda k: k["AbsY"], reverse=True)[0]
            
            peak_shift = round(abs(r_top["파장(nm)"] - s_top["파장(nm)"]), 2)
            intensity_ratio = min(r_top["AbsY"], s_top["AbsY"]) / max(r_top["AbsY"], s_top["AbsY"]) * 100
            
            shift_penalty = max(0, 100 - (peak_shift * 10))
            symmetry_score = round((intensity_ratio * 0.6) + (shift_penalty * 0.4), 1)
            symmetry_score = max(5.0, min(99.9, symmetry_score))
            
            symmetry_results.append({
                "실험 조건(그룹)": g_key,
                "R형 대표 파장": f"{r_top['파장(nm)']} nm ({r_top['유형']})",
                "S형 대표 파장": f"{s_top['파장(nm)']} nm ({s_top['유형']})",
                "파장 편차(Shift)": f"{peak_shift} nm",
                "거울상 대칭성 일치도(%)": f"{symmetry_score}%",
                "대칭성 평가": "완벽한 대칭 (Mirror-image)" if symmetry_score > 85 else ("보통" if symmetry_score > 60 else "대칭성 붕괴 (비대칭)")
            })
        else:
            symmetry_results.append({
                "실험 조건(그룹)": g_key,
                "R형 대표 파장": "데이터 있음" if r_peaks else "누락",
                "S형 대표 파장": "데이터 있음" if s_peaks else "누락",
                "파장 편차(Shift)": "-",
                "거울상 대칭성 일치도(%)": "측정 불가",
                "대칭성 평가": "R 또는 S 쌍(Pair) 불완전"
            })

    with data_tab1:
        st.pyplot(fig)
        if anomaly_reports:
            st.warning("### 🚨 이상 탐지(Anomaly Detection) 보고")
            for report in set(anomaly_reports):
                st.write(report)
        
    with data_tab2:
        st.subheader("📊 동일 조건 내 [R-form vs S-form] 거울상 대칭성 정량 분석")
        st.info("💡 카이랄 실험의 핵심은 **동일 조건(시간/개폐)에서 R형과 S형이 서로 완벽한 거울상 대칭(Cotton Effect 반전 및 일치)**을 이루는지 비교하는 것입니다.")
        if symmetry_results:
            sym_df = pd.DataFrame(symmetry_results)
            st.dataframe(sym_df, use_container_width=True)
        else:
            st.warning("비교할 수 있는 R/S 쌍 데이터가 없습니다.")

    with data_tab3:
        if peak_summary:
            peak_df = pd.DataFrame(peak_summary).drop(columns=["AbsY"], errors='ignore')
            peak_df = peak_df.sort_values(by=["조건그룹", "파장(nm)"]).reset_index(drop=True)
            st.dataframe(peak_df, use_container_width=True)
        else:
            st.info("조건에 맞는 피크가 없습니다.")

    with data_tab4:
        st.subheader("📂 저장된 분석 결과 기록 (Save Archive)")
        if st.session_state.saved_reports:
            for idx, item in enumerate(st.session_state.saved_reports):
                with st.expander(f"📌 [{idx+1}] 저장된 리포트 ({item['time']}) - 파일수: {item['file_count']}개"):
                    st.markdown(item['content'])
                    if st.button(f"🗑️ 이 기록 삭제 (No. {idx+1})", key=f"del_{idx}"):
                        st.session_state.saved_reports.pop(idx)
                        st.rerun()
            if st.button("🧹 모든 기록 초기화"):
                st.session_state.saved_reports.clear()
                st.rerun()
        else:
            st.info("💡 저장된 기록이 없습니다. 아래에서 리포트를 생성한 뒤 Save 버튼을 눌러보세요!")

    # ==========================================
    # 3. OpenAI API 연동: 정확한 R/S 비교 기반 논문형 리포트
    # ==========================================
    st.write("---")
    st.header("🤖 AI 수석 연구원: R/S 대칭성 비교 중심 논문형 리포트")
    st.write("R-form과 S-form 간의 거울상 대칭성 검증에 초점을 맞춘 Methods-to-Conclusion 리포트를 생성합니다.")
    
    if st.button("🚀 AI 논문형 리포트 생성"):
        if not api_key:
            st.error("⚠️ 좌측 사이드바에 OpenAI API Key를 입력해 주세요!")
        elif not peak_summary:
            st.warning("⚠️ 분석할 피크 데이터가 없습니다.")
        else:
            with st.spinner("AI가 R/S 이성질체 간 거울상 대칭성 비교를 중심으로 논문형 보고서를 작성 중입니다..."):
                try:
                    client = openai.OpenAI(api_key=api_key)
                    
                    prompt_data = pd.DataFrame(peak_summary).drop(columns=["AbsY"], errors='ignore').to_string(index=False)
                    sym_data = pd.DataFrame(symmetry_results).to_string(index=False)
                    anomaly_data = "\n".join(set(anomaly_reports)) if anomaly_reports else "탐지된 노이즈 없음. 양호."
                    meta_info = "\n".join(file_metadata)
                    
                    system_prompt = (
                        "당신은 화학 및 카이랄 분광학(CD) 분야의 세계적인 수석 연구원입니다. "
                        "이 실험의 본질은 **동일 조건(시간, 개폐 상태)에서 R-form과 S-form을 쌍(Pair)으로 비교하여 거울상 대칭성(Cotton Effect 및 부호 반전, 파장 일치도)을 평가**하는 것입니다. "
                        "절대로 개별 샘플(예: R_open_30 혼자서)의 대칭성이 높다는 식의 치명적인 오류를 범하지 마세요. 대칭성은 반드시 R형과 S형을 서로 비교할 때만 성립합니다.\n\n"
                        "다음 논문 구조에 맞춰 리포트를 작성하세요:\n"
                        "### 1. 🧪 Methods (실험 방법 개요)\n"
                        "- R-form과 S-form을 동일 조건별로 짝지어 비교하는 대칭성 검증 실험 구조 요약.\n\n"
                        "### 2. 📊 Results (R/S 대칭성 비교 결과)\n"
                        "- 동일 조건(시간/상태)별 R형과 S형의 거울상 대칭성 지표(%) 및 파장 이동량(Shift) 결과를 수치 기반으로 명확히 서술.\n"
                        "- 이상 탐지(Anomaly Detection) 데이터 반영.\n\n"
                        "### 3. 🧠 Discussion (거울상 대칭성 분석 및 물리화학적 의미)\n"
                        "- R/S 쌍 간의 대칭성이 유지되거나 붕괴된 원인 추론 (예: Sample alignment 오차, Film thickness 비균일성, 온도 편차 등).\n"
                        "- 카이랄성 발현 및 Molecular packing, Crystallinity 관점에서의 해석.\n\n"
                        "### 4. 🚀 Conclusion & Next Steps (결론 및 다음 실험 제안)\n"
                        "- 가장 완벽한 거울상 대칭성을 보여준 최적의 조건(시간 등) 결론 도출.\n"
                        "- 다음 실험을 위한 구체적 제안을 체크박스(- [ ]) 형태로 제시 (예: 110분 추가 측정, 온도 보정 등)."
                    )
                    
                    user_prompt = f"### 메타 데이터\n{meta_info}\n\n### R vs S 대칭성 정량 비교 데이터\n{sym_data}\n\n### 이상 탐지 내역\n{anomaly_data}\n\n### 핵심 피크 데이터\n{prompt_data}\n\n위 데이터를 바탕으로 R/S 쌍 비교 중심의 정확한 논문형 리포트를 작성해 줘."
                    
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.2
                    )
                    
                    report_content = response.choices[0].message.content
                    st.session_state.current_report = report_content
                    st.session_state.current_file_count = len(uploaded_files)
                    
                    st.success("✅ AI 논문형 리포트 생성 완료!")
                    
                except Exception as e:
                    st.error(f"OpenAI API 호출 중 오류가 발생했습니다: {e}")

    if 'current_report' in st.session_state:
        st.markdown("---")
        st.markdown(st.session_state.current_report)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 이 분석 결과 Save하기 (기록 보관함에 추가)"):
                from datetime import datetime
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.saved_reports.append({
                    "time": current_time,
                    "file_count": st.session_state.current_file_count,
                    "content": st.session_state.current_report
                })
                st.success("🎉 분석 결과가 [저장된 분석 기록] 탭에 성공적으로 Save되었습니다!")
        with col2:
            md_file = io.BytesIO(st.session_state.current_report.encode('utf-8'))
            st.download_button(
                label="📄 최종 리포트 다운로드 (.md / Word 호환)",
                data=md_file,
                file_name="Chiral_Spectroscopy_Report.md",
                mime="text/markdown"
            )
