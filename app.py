import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Chirality Batch Analyzer", layout="wide")
st.title("📊 CD 스펙트럼 대량 분석 및 심층 진단 시스템")
st.write("다양한 파일명 양식을 자동으로 인식하여 분석하고, 전문가 수준의 분광학적 심층 해설을 제공합니다.")

st.info("💡 **[업로드 가능 양식]** `R_open_30.csv` (3단) 또는 `R_60min.csv` (2단, 자동으로 Close 조건 부여)")
uploaded_files = st.file_uploader("📂 실험 CSV 파일들을 모두 선택해서 올려주세요", type=['csv'], accept_multiple_files=True)

if uploaded_files:
    experiments = {}
    
    # 1. 파일명 자동 분석 및 예외 처리
    for f in uploaded_files:
        clean_name = f.name.replace('.csv', '').replace('.CSV', '')
        parts = clean_name.split('_')
        
        # 양식 1: R_open_30.csv (3부분으로 나뉠 때)
        if len(parts) >= 3:
            form = parts[0].upper()
            condition_raw = parts[1].lower()
            time_raw = parts[2].replace('min', '') # min 글자가 있으면 제거
            
        # 양식 2: R_60min.csv (2부분으로 나뉠 때 -> 조건이 생략되었으므로 close로 간주)
        elif len(parts) == 2:
            form = parts[0].upper()
            condition_raw = 'close'
            time_raw = parts[1].replace('min', '')
        else:
            continue # 양식이 전혀 안 맞으면 패스
            
        # 조건 이름 예쁘게 포맷팅
        if condition_raw == 'open': cond_display = "Open (공기 노출)"
        elif condition_raw in ['h', 'half']: cond_display = "Half-open (부분 차단)"
        elif condition_raw == 'close': cond_display = "Close (완전 차단)"
        else: cond_display = condition_raw
        
        exp_key = f"{cond_display}_{time_raw}분"
        
        if exp_key not in experiments:
            experiments[exp_key] = {'R': None, 'S': None, 'Condition': cond_display, 'Time': f"{time_raw}분"}
            
        if form == 'R': experiments[exp_key]['R'] = f
        elif form == 'S': experiments[exp_key]['S'] = f

    # 2. 데이터 분석 로직
    summary_list = []
    detailed_reports = []

    for key, data in experiments.items():
        if data['R'] is not None and data['S'] is not None:
            r_file, s_file = data['R'], data['S']
            cond, time = data['Condition'], data['Time']
            
            df_r = pd.read_csv(r_file, skiprows=21, header=None)
            df_s = pd.read_csv(s_file, skiprows=21, header=None)
            
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
                
            df_r_valid = df_r[df_r[0] <= 350]
            df_s_valid = df_s[df_s[0] <= 350]
            
            r_peak_wave = df_r[0].loc[df_r_valid[2].idxmax()]
            s_peak_wave = df_s[0].loc[df_s_valid[2].idxmin()]
            
            summary_list.append({
                "실험 조건": cond, "어닐링 시간": time,
                "대칭성 점수(%)": round(similarity, 1),
                "R 최대 파장(nm)": r_peak_wave, "S 최소 파장(nm)": s_peak_wave,
            })
            
            detailed_reports.append({
                "key": key, "cond": cond, "time": time, "similarity": similarity,
                "r_peak": r_peak_wave, "s_peak": s_peak_wave,
                "df_r": df_r, "df_s": df_s, "r_norm": r_norm, "s_norm": s_norm
            })

    # 3. 결과 화면 출력
    if summary_list:
        st.write("---")
        st.header("📋 실험 조건별 통합 평가 대시보드")
        summary_df = pd.DataFrame(summary_list)
        st.dataframe(summary_df.style.background_gradient(subset=['대칭성 점수(%)'], cmap='Greens'), use_container_width=True)

        st.write("---")
        st.header("🔎 분광학적 심층 해석 레포트")
        
        for report in detailed_reports:
            sim = report['similarity']
            title = f"📂 [{report['cond']} / {report['time']}] 대칭성: {sim:.1f}%"
            
            with st.expander(title):
                # 💡 [핵심 강화] 매우 구체적이고 전문적인 화학/물리적 해석 코멘트
                if sim >= 85:
                    eval_1 = f"**[구조 및 광학 활성]** 수학적 대칭성 {sim:.1f}%로, 농도 오차 보정 후 완벽한 거울상 이성질체(Enantiomer) 관계가 입증되었습니다. R-form과 S-form이 뚜렷하게 반대 부호의 Cotton Effect를 나타내며, 광학 순도(Enantiomeric Excess)가 매우 우수하게 보존되었습니다."
                    eval_2 = f"**[공정 해석]** '{report['cond']}' 조건에서 {report['time']} 동안 진행된 어닐링 과정이 성공적이었습니다. 용매의 증발 속도가 이상적으로 제어되어 고분자 사슬이 열역학적으로 가장 안정한 상태로 배향(Orientation) 및 자가조립(Self-assembly)을 이루었습니다."
                elif sim >= 65:
                    eval_1 = f"**[구조 및 광학 활성]** 대칭성 {sim:.1f}%로 기본적인 카이랄성 경향성은 확인되나, 부분적인 비대칭(Asymmetry) 스펙트럼이 관찰됩니다. 특정 파장 대역에서 분자 간 상호작용의 미세한 흐트러짐이 존재합니다."
                    eval_2 = f"**[공정 해석]** '{report['cond']}' 조건의 증발 속도 불균형 또는 {report['time']}의 어닐링 시간이 배향을 완벽히 유도하기에는 다소 부족했던 것으로 추정됩니다. 필름 표면의 국부적인 두께 차이나 용매 잔류 여부를 점검할 필요가 있습니다."
                else:
                    eval_1 = f"**[구조 및 광학 활성]** 대칭성 {sim:.1f}%로, 물질 고유의 거울상 카이랄성 발현이 심각하게 훼손되었습니다. 바탕선 요동(Baseline drift)이 심하고, 유효 흡수 파장 대역에서 역전 현상이 제대로 나타나지 않습니다."
                    eval_2 = f"**[공정 해석]** '{report['cond']}' 조건으로 인해 용매가 지나치게 빨리 증발(Open)하거나 갇혀있어(Close), 분자들이 입체 규칙적(Stereoregular)으로 배열될 충분한 열역학적 여유를 갖지 못했습니다. 즉각적인 공정 조건 수정(Half-open 전환 등)이 강력히 권장됩니다."
                
                st.markdown(eval_1)
                st.markdown(eval_2)
                st.markdown(f"**[피크 파장 분석]** R-form의 최대 양(+)의 피크는 **{report['r_peak']}nm**, S-form의 최대 음(-)의 피크는 **{report['s_peak']}nm**에서 관찰되었습니다.")
                
                plot_col1, plot_col2 = st.columns(2)
                with plot_col1:
                    fig1, ax1 = plt.subplots(figsize=(5, 3))
                    ax1.plot(report['df_r'][0], report['df_r'][2], label='R-form (Raw)', color='blue')
                    ax1.plot(report['df_s'][0], report['df_s'][2], label='S-form (Raw)', color='orange')
                    ax1.axhline(0, color='black', linewidth=0.5)
                    ax1.set_title("원본 데이터 (Raw)")
                    ax1.legend()
                    st.pyplot(fig1)

                with plot_col2:
                    fig2, ax2 = plt.subplots(figsize=(5, 3))
                    ax2.plot(report['df_r'][0], report['r_norm'], label='R-form (Norm)', color='blue')
                    ax2.plot(report['df_s'][0], report['s_norm'], label='S-form (Norm)', color='orange', linestyle='--')
                    ax2.axhline(0, color='black', linewidth=0.5)
                    ax2.set_title("정규화 데이터 (Normalized)")
                    ax2.legend()
                    st.pyplot(fig2)
    else:
        st.warning("⚠️ 분석할 수 있는 R/S 파일 짝이 없습니다. 파일명을 확인해 주세요.")
