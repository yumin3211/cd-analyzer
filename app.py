import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import openai
import io

# ==========================================
# 웹사이트 기본 설정
# ==========================================
st.set_page_config(page_title="AI CD Spectrum Analyzer", layout="wide")

st.title("🔬 AI CD Spectrum Analyzer")
st.write("다중 샘플 간의 거울상 대칭성(Mirror Symmetry) 정량 비교, 피크 이동량 분석 및 ACS 논문 수준의 심층 리포트를 자동 생성합니다.")

if 'saved_reports' not in st.session_state:
    st.session_state.saved_reports = []

# ==========================================
# 1. 사이드바: 분석 설정 및 OpenAI API
# ==========================================
with st.sidebar:
    st.header("⚙️ 분석 및 AI 설정")
    api_key = st.text_input("🔑 OpenAI API Key 입력", type="password")
    
    st.write("---")
    spec_type = st.selectbox("📊 분광 데이터 종류", ["원편광이색성 (CD)", "자외선-가시광선 (UV-Vis)", "적외선 (FT-IR)", "라만 (Raman)"])
    
    st.write("---")
    st.write("🔍 피크(Peak) 감지 설정")
    prominence = st.slider("피크 감지 민감도 (Prominence)", min_value=0.1, max_value=3.0, value=0.5, step=0.1)
    max_peaks_to_show = st.slider("핵심 피크 표시 개수 제한", min_value=1, max_value=10, value=5)

# ==========================================
# 2. 데이터 업로드 및 스마트 파싱
# ==========================================
uploaded_files = st.file_uploader("📂 R/S form 및 조건별 CSV 파일 업로드", type=['csv'], accept_multiple_files=True)

if uploaded_files:
    st.write("---")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    peak_summary = []
    file_metadata = []
    time_groups = {}
    
    # 탭 구성 (랭킹 및 비교 탭 추가)
    tab_graph, tab_ranking, tab_raw_data, tab_archive = st.tabs([
        "📈 논문형 오버레이 그래프", 
        "🏆 종합 랭킹 & Batch Compare", 
        "📋 정량 데이터 & Peak 요약",
        "📂 분석 기록 Archive"
    ])
    
    for f in uploaded_files:
        try:
            clean_name = f.name.replace('.csv', '').replace('.CSV', '')
            parts = clean_name.split('_')
            
            form = "UNKNOWN"
            if len(parts) > 0:
                if parts[0].upper() in ["R", "S"]: form = parts[0].upper()
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
            
            group_key = f"{time_val}_{condition}"
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
            
            ax.plot(x, y, label=f.name, linewidth=1.5, alpha=0.8)
            ax.plot(x[peaks_pos], y[peaks_pos], "x", color='red', markersize=5)
            ax.plot(x[peaks_neg], y[peaks_neg], "x", color='blue', markersize=5)
            
            sample_peaks = []
            for p in peaks_pos:
                sample_peaks.append({"샘플명": f.name, "형태": form, "조건그룹": group_key, "유형": "Positive", "파장(nm)": round(x[p], 2), "강도(Y)": round(y[p], 3), "AbsY": abs(y[p])})
            for p in peaks_neg:
                sample_peaks.append({"샘플명": f.name, "형태": form, "조건그룹": group_key, "유형": "Negative", "파장(nm)": round(x[p], 2), "강도(Y)": round(y[p], 3), "AbsY": abs(y[p])})
            
            if group_key not in time_groups:
                time_groups[group_key] = {"R": [], "S": []}
            if form == "R": time_groups[group_key]["R"].extend(sample_peaks)
            elif form == "S": time_groups[group_key]["S"].extend(sample_peaks)

            sample_peaks = sorted(sample_peaks, key=lambda k: k["AbsY"], reverse=True)[:max_peaks_to_show]
            peak_summary.extend(sample_peaks)
                
        except Exception as e:
            st.error(f"{f.name} 처리 중 오류 발생: {e}")

    # ==========================================
    # 정량적 대칭성 및 랭킹 데이터 산출 (Python Base)
    # ==========================================
    raw_calc_results = []
    
    for g_key, forms in time_groups.items():
        r_peaks = forms["R"]
        s_peaks = forms["S"]
        
        if r_peaks and s_peaks:
            r_top = sorted(r_peaks, key=lambda k: k["AbsY"], reverse=True)[0]
            s_top = sorted(s_peaks, key=lambda k: k["AbsY"], reverse=True)[0]
            
            shift_val = round(abs(r_top["파장(nm)"] - s_top["파장(nm)"]), 2)
            intensity_ratio = min(r_top["AbsY"], s_top["AbsY"]) / max(r_top["AbsY"], s_top["AbsY"]) * 100
            
            shift_penalty = max(0, 100 - (shift_val * 8))
            sym_val = round((intensity_ratio * 0.6) + (shift_penalty * 0.4), 1)
            sym_val = max(5.0, min(99.9, sym_val))
            
            # 논문형 그래프 자동 표시 (어노테이션)
            anchor_x = r_top["파장(nm)"]
            anchor_y = r_top["강도(Y)"]
            ax.annotate(f"[{g_key}]\nShift: {shift_val} nm\nSym: {sym_val}%",
                        xy=(anchor_x, anchor_y), xytext=(15, 15),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.9),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color="gray"),
                        fontsize=9)
            
            raw_calc_results.append({
                "Group": g_key,
                "Shift_nm": shift_val,
                "Symmetry_%": sym_val,
                "R_Peak": r_top["파장(nm)"],
                "S_Peak": s_top["파장(nm)"]
            })

    ax.axhline(0, color='black', linewidth=0.8)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    with tab_graph:
        st.pyplot(fig)
        st.caption("※ 그래프 위에 R/S 짝이 맞는 조건의 Peak Shift 및 Mirror Symmetry 값이 자동 표시됩니다.")

    with tab_ranking:
        if raw_calc_results:
            st.subheader("🏆 조건별 분석 랭킹 및 Batch Compare")
            df_calc = pd.DataFrame(raw_calc_results)
            
            col1, col2, col3 = st.columns(3)
            
            # 1. Symmetry Ranking
            with col1:
                st.markdown("#### 🥇 Symmetry Ranking")
                sym_sorted = df_calc.sort_values(by="Symmetry_%", ascending=False)
                for idx, row in sym_sorted.iterrows():
                    st.markdown(f"**{row['Group']}** : {row['Symmetry_%']}%")
            
            # 2. Shift Ranking (낮을수록 좋음)
            with col2:
                st.markdown("#### 🎯 Peak Shift Ranking")
                shift_sorted = df_calc.sort_values(by="Shift_nm", ascending=True)
                for idx, row in shift_sorted.iterrows():
                    st.markdown(f"**{row['Group']}** : {row['Shift_nm']} nm")
                    
            # 3. Batch Compare (Best/Worst)
            with col3:
                st.markdown("#### 🏭 Batch Compare")
                best_sample = sym_sorted.iloc[0]['Group']
                worst_sample = sym_sorted.iloc[-1]['Group']
                st.markdown(f"- **Best Sample:** `{best_sample}`")
                st.markdown(f"- **Worst Sample:** `{worst_sample}`")
                st.markdown(f"- **Recommendation:**\n  `{best_sample}` 공정 유지\n  `{worst_sample}` Annealing 조건 재검토")
        else:
            st.warning("대칭성을 비교할 수 있는 R/S 짝이 부족하여 랭킹을 산출할 수 없습니다.")

    with tab_raw_data:
        if raw_calc_results:
            st.dataframe(pd.DataFrame(raw_calc_results).style.format({"Shift_nm": "{:.2f}", "Symmetry_%": "{:.1f}"}), use_container_width=True)
        if peak_summary:
            df_peaks = pd.DataFrame(peak_summary).drop(columns=["AbsY"], errors='ignore')
            st.dataframe(df_peaks.sort_values(by=["조건그룹", "파장(nm)"]).reset_index(drop=True), use_container_width=True)

    with tab_archive:
        st.subheader("📂 저장된 분석 결과 기록 (Save Archive)")
        if st.session_state.saved_reports:
            for idx, item in enumerate(st.session_state.saved_reports):
                with st.expander(f"📌 [{idx+1}] 저장된 리포트 ({item['time']})"):
                    st.markdown(item['content'])
                    if st.button(f"🗑️ 이 기록 삭제", key=f"del_{idx}"):
                        st.session_state.saved_reports.pop(idx)
                        st.rerun()
        else:
            st.info("아직 저장된 기록이 없습니다.")

    # ==========================================
    # 3. AI CD Spectrum Analyzer 프롬프트 및 리포트 생성
    # ==========================================
    st.write("---")
    st.header("🤖 AI CD Spectrum Analyzer 심층 리포트")
    
    col_btn1, col_btn2 = st.columns(2)
    
    prompt_data = pd.DataFrame(peak_summary).drop(columns=["AbsY"], errors='ignore').to_string(index=False)
    calc_data = pd.DataFrame(raw_calc_results).to_string(index=False) if raw_calc_results else "비교 데이터 없음"
    meta_info = "\n".join(file_metadata)

    # 버튼 1: AI 종합 비교 리포트 생성
    if col_btn1.button("📊 종합 비교 리포트 생성 (Confidence & Error 포함)"):
        if not api_key: st.error("⚠️ OpenAI API Key를 입력해 주세요!")
        elif not raw_calc_results: st.warning("⚠️ R/S 비교 데이터가 필요합니다.")
        else:
            with st.spinner("AI가 데이터 신뢰도 및 비교 분석 리포트를 작성 중입니다..."):
                try:
                    client = openai.OpenAI(api_key=api_key)
                    system_prompt = (
                        "당신은 AI CD Spectrum Analyzer입니다. 데이터를 단순 나열하지 말고, 조건 간의 수치적 비교를 통해 날카롭게 분석하세요. "
                        "반드시 아래의 마크다운 포맷을 그대로 사용하여 작성하세요:\n\n"
                        "### 1. AI Confidence\n"
                        "★★★★☆ (별점으로 신뢰도 표시)\n"
                        "- Peak quality : High / Medium / Low\n"
                        "- Noise : Low / High\n"
                        "- Matching confidence : [수치]%\n\n"
                        "### 2. Data Reliability\n"
                        "[수치]% \n"
                        "**Reason:** (Peak count, Noise, Shift, Intensity 측면에서 근거 제시)\n\n"
                        "### 3. Possible Error\n"
                        "- Sample thickness : [★ 개수로 점수화]\n"
                        "- Baseline : [★ 개수로 점수화]\n"
                        "- Instrument Noise : [★ 개수로 점수화]\n"
                        "- Polarizer : [★ 개수로 점수화]\n\n"
                        "### 4. AI Discussion (조건 간 구체적 비교)\n"
                        "(예시 포맷에 맞춰 정확한 데이터 수치로 비교 서술)\n"
                        "A 조건에서는 Shift가 [X] nm로 가장 작고 Mirror symmetry score가 [Y]%로 가장 높았다.\n"
                        "이는 Annealing 동안 분자의 chiral arrangement가 안정적으로 유지되었음을 시사한다.\n"
                        "반면 B 조건에서는 Shift가 [Z] nm로 증가하였으며, Positive/Negative peak pairing이 깨져 분자배향 불균일 가능성이 높다.\n"
                        "결과적으로 A 조건이 B 조건보다 Mirror symmetry가 [개선율]% 향상되었다."
                    )
                    user_prompt = f"### 메타 데이터\n{meta_info}\n\n### 대칭성 및 Shift 계산 결과\n{calc_data}\n\n위 데이터를 바탕으로 지정된 포맷의 분석 리포트를 작성해 줘."
                    
                    response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.2)
                    st.session_state.current_report = response.choices[0].message.content
                    st.rerun()
                except Exception as e:
                    st.error(f"오류 발생: {e}")

    # 버튼 2: ACS 수준 Conclusion 생성
    if col_btn2.button("📝 Generate Research Conclusion (ACS Paper 수준)"):
        if not api_key: st.error("⚠️ OpenAI API Key를 입력해 주세요!")
        elif not raw_calc_results: st.warning("⚠️ R/S 비교 데이터가 필요합니다.")
        else:
            with st.spinner("ACS Applied Materials 수준의 Conclusion을 작성 중입니다..."):
                try:
                    client = openai.OpenAI(api_key=api_key)
                    system_prompt = (
                        "당신은 AI CD Spectrum Analyzer입니다. 사용자의 분광 데이터를 기반으로 "
                        "ACS Applied Materials 저널에 실릴 수준의 최고급 Research Conclusion을 마크다운으로 작성하세요.\n\n"
                        "**작성 규칙:**\n"
                        "1. '본 결과는 [가장 좋은 조건]에서 분자배향 안정성이 가장 우수함을 보여준다.' 형식으로 시작.\n"
                        "2. 데이터(Symmetry %, Shift nm)를 근거로 Film packing, Crystallinity 향상 등을 물리화학적으로 논증.\n"
                        "3. 마지막 단락에 '향후에는 Annealing time과 Temperature를 추가 변수로 고려할 필요가 있다.' 등 명확한 후속 연구 제언 포함."
                    )
                    user_prompt = f"### 대칭성 및 Shift 계산 결과\n{calc_data}\n\n### 핵심 피크 데이터\n{prompt_data}\n\n최고급 저널 수준의 Conclusion을 작성해 줘."
                    
                    response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.2)
                    st.session_state.current_report = "### 🎓 ACS-Level Research Conclusion\n\n" + response.choices[0].message.content
                    st.rerun()
                except Exception as e:
                    st.error(f"오류 발생: {e}")

    # 리포트 출력 및 Save 기능
    if 'current_report' in st.session_state:
        st.markdown("---")
        st.markdown(st.session_state.current_report)
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if st.button("💾 현재 분석 결과 Save (아카이브에 저장)"):
                from datetime import datetime
                st.session_state.saved_reports.append({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "content": st.session_state.current_report
                })
                st.success("🎉 [분석 기록 Archive] 탭에 성공적으로 저장되었습니다!")
        with col_s2:
            md_file = io.BytesIO(st.session_state.current_report.encode('utf-8'))
            st.download_button(label="📄 Markdown 파일 다운로드 (Word 호환)", data=md_file, file_name="AI_CD_Spectrum_Report.md", mime="text/markdown")
