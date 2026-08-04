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
# 0. 세션 상태(Session State) 초기화 (저장 기능용)
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
    st.header("📊 스펙트럼 오버레이 및 정량적 카이랄 대칭성 분석")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    peak_summary = []
    file_metadata = []
    time_groups = {}
    anomaly_reports = []
    
    data_tab1, data_tab2, data_tab3, data_tab4 = st.tabs([
        "📈 오버레이 스펙트럼", 
        "📊 시간대별 정량 비교(대칭성 및 이동량)", 
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
                anomaly_reports.append(f"⚠️ **{f.name}**: 비정상적으로 많은 피크({total_peaks_detected}개)가 검출되었습니다. Instrument noise가 의심되므로 재측정(Repeat measurement)을 권장합니다.")
            if len(y) > 0 and max(abs(y)) < 0.5:
                anomaly_reports.append(f"⚠️ **{f.name}**: 최대 신호 세기가 매우 낮습니다(Max < 0.5). Sample concentration 부족 또는 Film thickness 문제를 확인하세요.")
            
            ax.plot(x, y, label=f.name, linewidth=1.5)
            ax.plot(x[peaks_pos], y[peaks_pos], "x", color='red', markersize=6)
            ax.plot(x[peaks_neg], y[peaks_neg], "x", color='blue', markersize=6)
            
            sample_peaks = []
            for p in peaks_pos:
                sample_peaks.append({"샘플명": f.name, "형태": form, "그룹": group_key, "유형": "Positive", "파장(nm)": round(x[p], 2), "강도(Y)": round(y[p], 3), "AbsY": abs(y[p])})
            for p in peaks_neg:
                sample_peaks.append({"샘플명": f.name, "형태": form, "그룹": group_key, "유형": "Negative", "파장(nm)": round(x[p], 2), "강도(Y)": round(y[p], 3), "AbsY": abs(y[p])})
            
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
    
    symmetry_results = []
    for g_key, forms in time_groups.items():
        r_peaks = sorted(forms["R"], key=lambda k: k["AbsY"], reverse=True)
        s_peaks = sorted(forms["S"], key=lambda k: k["AbsY"], reverse=True)
        
        if r_peaks and s_peaks:
            r_max_peak = r_peaks[0]
            s_max_peak = s_peaks[0]
            
            peak_shift = round(abs(r_max_peak["파장(nm)"] - s_max_peak["파장(nm)"]), 2)
            
            r_intensity = r_max_peak["AbsY"]
            s_intensity = s_max_peak["AbsY"]
            intensity_ratio = min(r_intensity, s_intensity) / max(r_intensity, s_intensity) * 100
            
            shift_penalty = max(0, 100 - (peak_shift * 10))
            symmetry_score = round((intensity_ratio * 0.6) + (shift_penalty * 0.4), 1)
            symmetry_score = max(5.0, min(99.9, symmetry_score))
            
            symmetry_results.append({
                "조건(시간)": g_key,
                "R형 주요 파장(nm)": r_max_peak["파장(nm)"],
                "S형 주요 파장(nm)": s_max_peak["파장(nm)"],
                "피크 이동량(Shift)": f"{peak_shift} nm",
                "거울상 대칭성(%)": f"{symmetry_score}%",
                "평가": "우수" if symmetry_score > 85 else ("보통" if symmetry_score > 60 else "비대칭 심화 (검토 요망)")
            })
        elif r_peaks or s_peaks:
            symmetry_results.append({
                "조건(시간)": g_key,
                "R형 주요 파장(nm)": r_peaks[0]["파장(nm)"] if r_peaks else "-",
                "S형 주요 파장(nm)": s_peaks[0]["파장(nm)"] if s_peaks else "-",
                "피크 이동량(Shift)": "-",
                "거울상 대칭성(%)": "측정 불가",
                "평가": "비교 대상(R/S 짝) 누락"
            })

    with data_tab1:
        st.pyplot(fig)
        if anomaly_reports:
            st.warning("### 🚨 이상 탐지(Anomaly Detection) 보고")
            for report in set(anomaly_reports):
                st.write(report)
        
    with data_tab2:
        st.subheader("📊 동일 조건/시간 내 R-form vs S-form 정량 비교")
        if symmetry_results:
            sym_df = pd.DataFrame(symmetry_results)
            st.dataframe(sym_df, use_container_width=True)
        else:
            st.warning("대칭성을 비교할 수 있는 짝(R/S) 데이터가 없습니다.")

    with data_tab3:
        if peak_summary:
            peak_df = pd.DataFrame(peak_summary).drop(columns=["AbsY"], errors='ignore')
            peak_df = peak_df.sort_values(by=["그룹", "파장(nm)"]).reset_index(drop=True)
            st.dataframe(peak_df, use_container_width=True)
        else:
            st.info("조건에 맞는 피크가 없습니다.")

    with data_tab4:
        st.subheader("📂 저장된 분석 결과 기록 (Save Archive)")
        if st.session_state.saved_reports:
            for idx, item in enumerate(st.session_state.saved_reports):
                with st.expander(f"📌 [{idx+1}] 저장된 리포트 ({item['time']}) - 업로드 파일수: {item['file_count']}개"):
                    st.markdown(item['content'])
                    if st.button(f"🗑️ 이 기록 삭제하기 (No. {idx+1})", key=f"del_{idx}"):
                        st.session_state.saved_reports.pop(idx)
                        st.rerun()
            if st.button("🧹 모든 저장 기록 초기화"):
                st.session_state.saved_reports.clear()
                st.rerun()
        else:
            st.info("💡 아직 저장된 분석 기록이 없습니다. 아래에서 AI 리포트를 생성한 뒤 **[💾 이 분석 결과 Save하기]** 버튼을 눌러보세요!")

    # ==========================================
    # 3. OpenAI API 연동 및 Save 기능
    # ==========================================
    st.write("---")
    st.header("🤖 AI 수석 연구원: 논문형 심층 분석 및 후속 제안 리포트")
    st.write("Methods, Results, Discussion, Conclusion 형식의 완벽한 논문 구조 리포트 생성 및 저장 시스템입니다.")
    
    if st.button("🚀 AI 논문형 리포트 생성"):
        if not api_key:
            st.error("⚠️ 좌측 사이드바에 OpenAI API Key를 입력해 주세요!")
        elif not peak_summary:
            st.warning("⚠️ 분석할 피크 데이터가 없습니다.")
        else:
            with st.spinner("AI가 논문형 리포트를 작성 중입니다..."):
                try:
                    client = openai.OpenAI(api_key=api_key)
                    
                    prompt_data = pd.DataFrame(peak_summary).drop(columns=["AbsY"], errors='ignore').to_string(index=False)
                    sym_data = pd.DataFrame(symmetry_results).to_string(index=False)
                    anomaly_data = "\n".join(set(anomaly_reports)) if anomaly_reports else "탐지된 특이 노이즈 없음. 데이터 양호."
                    meta_info = "\n".join(file_metadata)
                    
                    system_prompt = (
                        "당신은 화학 및 카이랄 분광학 분야의 세계적인 수석 연구원입니다. "
                        "사용자의 분광 데이터를 바탕으로, 단순 요약이 아닌 '결과 해석, 원인 추론, 실험의 물리화학적 의미 설명'이 포함된 최고 수준의 논문형 보고서를 작성해 주세요. "
                        "다음 구조(Methods, Results, Discussion, Conclusion & Next Steps)를 엄격히 지켜 마크다운으로 작성하세요.\n\n"
                        "### 1. 🧪 Methods (실험 방법 개요)\n"
                        "- 업로드된 파일 정보와 조건(Annealing 시간, 개폐 여부)을 바탕으로 실험이 어떻게 구성되었는지 간략히 서술.\n\n"
                        "### 2. 📊 Results (결과 및 정량 지표)\n"
                        "- 시간별 R/S형의 거울상 대칭성(%), 피크 이동량(Shift), 그리고 강도 변화 트렌드를 수치에 기반하여 명확히 서술.\n"
                        "- 전달된 '이상 탐지(Anomaly Detection)' 데이터가 있다면 노이즈나 비정상 피크 발생 여부를 서술.\n\n"
                        "### 3. 🧠 Discussion (심층 원인 분석 및 해석)\n"
                        "- Peak Shift 또는 대칭성 붕괴의 원인 추론 (예: Sample alignment, Film thickness variation, Temperature deviation 등).\n"
                        "- 카이랄성(CD signal) 증감의 열역학적/구조적 의미 설명 (예: Molecular packing, Crystallinity 증가, 배향 안정화 등).\n\n"
                        "### 4. 🚀 Conclusion & Next Steps (결론 및 다음 실험 제안)\n"
                        "- 현재 조건 중 가장 최적화된 조건(시간 등)을 결론 내림.\n"
                        "- 다음 실험 제안을 위해 **구체적인 시간, 온도, 농도 조건**을 체크박스(- [ ]) 형태로 제시 (예: 110분 추가 측정, 온도 5도 상향 조정 등)."
                    )
                    
                    user_prompt = f"### 메타 데이터\n{meta_info}\n\n### 대칭성 및 Shift 정량 데이터\n{sym_data}\n\n### 이상 탐지 내역\n{anomaly_data}\n\n### 핵심 피크 데이터\n{prompt_data}\n\n위 데이터를 바탕으로 완벽한 논문 구조의 리포트를 작성해 줘."
                    
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.2
                    )
                    
                    report_content = response.choices[0].message.content
                    
                    # 세션에 임시 저장용으로 보관
                    st.session_state.current_report = report_content
                    st.session_state.current_file_count = len(uploaded_files)
                    
                    st.success("✅ AI 논문형 리포트 생성 완료!")
                    
                except Exception as e:
                    st.error(f"OpenAI API 호출 중 오류가 발생했습니다: {e}")

    # 리포트가 생성되어 있는 경우에만 Save 및 다운로드 버튼 활성화
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
