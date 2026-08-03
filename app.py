import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 웹사이트 기본 설정
st.set_page_config(page_title="Data Analytics Portfolio", layout="wide")

st.title("🚀 통합 데이터 분석 및 이상치 탐지 플랫폼")
st.write("소재의 특성 비교 검증부터 범용 통계 분석까지 지원하는 하이브리드 대시보드입니다.")

# 두 가지 기능을 탭(Tab)으로 분리
tab1, tab2 = st.tabs(["🧪 소재 특성 비교 (카이랄성 검증)", "📈 범용 통계 분석 (이상치 탐지)"])

# ==========================================
# [탭 1] 소재 대칭성 비교 (그래프 및 심층 분석 복구 완료!)
# ==========================================
with tab1:
    st.header("🧪 소재 특성 및 대칭성 비교 검증 (A/B Test)")
    st.write("R-form과 S-form 데이터를 비교하여 거울상 이성질체 특성을 심층 분석합니다.")
    
    st.info("💡 **[업로드 양식]** `R_open_30.csv` 또는 `R_60min.csv` 형식의 파일을 짝맞춰 올려주세요.")
    uploaded_files = st.file_uploader("📂 비교할 CSV 파일들을 모두 선택해서 올려주세요", type=['csv'], accept_multiple_files=True, key="tab1_upload")

    if uploaded_files:
        experiments = {}
        # 파일명 분석 및 짝맞추기
        for f in uploaded_files:
            clean_name = f.name.replace('.csv', '').replace('.CSV', '')
            parts = clean_name.split('_')
            
            if len(parts) >= 3:
                form, condition_raw, time_raw = parts[0].upper(), parts[1].lower(), parts[2].replace('min', '')
            elif len(parts) == 2:
                form, condition_raw, time_raw = parts[0].upper(), 'close', parts[1].replace('min', '')
            else:
                continue
                
            if condition_raw == 'open': cond_display = "Open (공기 노출)"
            elif condition_raw in ['h', 'half']: cond_display = "Half-open (부분 차단)"
            elif condition_raw == 'close': cond_display = "Close (완전 차단)"
            else: cond_display = condition_raw
            
            exp_key = f"{cond_display}_{time_raw}분"
            if exp_key not in experiments:
                experiments[exp_key] = {'R': None, 'S': None, 'Condition': cond_display, 'Time': f"{time_raw}분"}
                
            if form == 'R': experiments[exp_key]['R'] = f
            elif form == 'S': experiments[exp_key]['S'] = f

        summary_list = []
        detailed_reports = [] # 💡 사라졌던 심층 레포트 데이터를 다시 저장합니다!

        # 데이터 분석 진행
        for key, data in experiments.items():
            if data['R'] is not None and data['S'] is not None:
                df_r = pd.read_csv(data['R'], skiprows=21, header=None)
                df_s = pd.read_csv(data['S'], skiprows=21, header=None)
                
                df_r[0], df_r[2] = pd.to_numeric(df_r[0], errors='coerce'), pd.to_numeric(df_r[2], errors='coerce')
                df_s[0], df_s[2] = pd.to_numeric(df_s[0], errors='coerce'), pd.to_numeric(df_s[2], errors='coerce')
                df_r, df_s = df_r.dropna(subset=[0, 2]), df_s.dropna(subset=[0, 2])
                
                r_norm = df_r[2] / df_r[2].abs().max()
                s_norm = df_s[2] / df_s[2].abs().max()
                
                try: similarity = np.corrcoef(r_norm, -s_norm)[0, 1] * 100
                except: similarity = 0
                
                # 피크 파장 추출 (유효 구간 350nm 이하)
                df_r_valid = df_r[df_r[0] <= 350]
                df_s_valid = df_s[df_s[0] <= 350]
                r_peak_wave = df_r[0].loc[df_r_valid[2].idxmax()]
                s_peak_wave = df_s[0].loc[df_s_valid[2].idxmin()]
                
                summary_list.append({
                    "실험 조건": data['Condition'], "어닐링 시간": data['Time'],
                    "대칭성 점수(%)": round(similarity, 1)
                })
                
                detailed_reports.append({
                    "cond": data['Condition'], "time": data['Time'], "similarity": similarity,
                    "r_peak": r_peak_wave, "s_peak": s_peak_wave,
                    "df_r": df_r, "df_s": df_s, "r_norm": r_norm, "s_norm": s_norm
                })

        # 화면 출력부 (표 + 💡심층 그래프 해설 복구)
        if summary_list:
            st.write("### 📋 분석 결과 요약 (대시보드)")
            summary_df = pd.DataFrame(summary_list)
            st.dataframe(summary_df.style.background_gradient(subset=['대칭성 점수(%)'], cmap='Greens'), use_container_width=True)

            st.write("---")
            st.header("🔎 분광학적 심층 해석 및 시각화 레포트")
            
            for report in detailed_reports:
                sim = report['similarity']
                title = f"📂 [{report['cond']} / {report['time']}] 대칭성: {sim:.1f}%"
                
                with st.expander(title):
                    if sim >= 85:
                        eval_1 = f"**[구조 및 광학 활성]** 수학적 대칭성 {sim:.1f}%로, 농도 오차 보정 후 완벽한 거울상 이성질체(Enantiomer) 관계가 입증되었습니다. R-form과 S-form이 뚜렷하게 반대 부호의 Cotton Effect를 나타냅니다."
                        eval_2 = f"**[공정 해석]** '{report['cond']}' 조건에서 {report['time']} 동안 진행된 어닐링 과정이 성공적이었습니다. 용매의 증발 속도가 이상적으로 제어되어 고분자 사슬이 안정한 상태로 배향되었습니다."
                    elif sim >= 65:
                        eval_1 = f"**[구조 및 광학 활성]** 대칭성 {sim:.1f}%로 기본적인 카이랄성 경향성은 확인되나, 부분적인 비대칭(Asymmetry) 스펙트럼이 관찰됩니다."
                        eval_2 = f"**[공정 해석]** '{report['cond']}' 조건의 증발 속도 불균형 또는 {report['time']}의 어닐링 시간이 배향을 완벽히 유도하기에는 다소 부족했던 것으로 추정됩니다. 추가적인 공정 최적화가 요구됩니다."
                    else:
                        eval_1 = f"**[구조 및 광학 활성]** 대칭성 {sim:.1f}%로, 물질 고유의 거울상 카이랄성 발현이 심각하게 훼손되었습니다."
                        eval_2 = f"**[공정 해석]** '{report['cond']}' 조건으로 인해 분자들이 입체 규칙적으로 배열될 열역학적 여유를 갖지 못했습니다. 즉각적인 공정 조건 수정(Half-open 전환 등)이 권장됩니다."
                    
                    st.markdown(eval_1)
                    st.markdown(eval_2)
                    st.markdown(f"**[피크 파장 분석]** R-form의 최고점(Max)은 **{report['r_peak']}nm**, S-form의 최저점(Min)은 **{report['s_peak']}nm**에서 관찰되었습니다.")
                    
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

# ==========================================
# [탭 2] 어떤 데이터든 다루는 범용 이상치 탐지 모드
# ==========================================
with tab2:
    st.header("📈 범용 통계 분석 및 이상치 탐지 (Anomaly Detection)")
    st.write("어떤 형태의 CSV 파일이든 변수를 자동 인식하여, 공정 이탈 및 불량 구간을 통계적으로 탐지합니다.")
    
    univ_file = st.file_uploader("📂 아무 CSV 데이터나 올려주세요", type=['csv'], key="tab2_upload")

    if univ_file:
        try:
            df_univ = pd.read_csv(univ_file)
            st.success("✅ 데이터 구조 파악 완료!")
            
            columns = df_univ.columns.tolist()
            col1, col2, col3 = st.columns(3)
            with col1: x_axis = st.selectbox("가로축(X축) 선택:", columns)
            with col2: y_axis = st.selectbox("세로축(Y축) 선택:", columns)
            with col3: threshold = st.slider("🚨 이상치 탐지 민감도 (Z-Score)", 1.0, 5.0, 2.5, 0.1)

            if pd.api.types.is_numeric_dtype(df_univ[x_axis]) and pd.api.types.is_numeric_dtype(df_univ[y_axis]):
                plot_df = df_univ[[x_axis, y_axis]].dropna()
                mean_y, std_y = plot_df[y_axis].mean(), plot_df[y_axis].std()
                plot_df['Z_score'] = (plot_df[y_axis] - mean_y) / std_y
                anomalies = plot_df[plot_df['Z_score'].abs() > threshold]

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(plot_df[x_axis], plot_df[y_axis], label='정상 데이터', color='royalblue', linewidth=1)
                if not anomalies.empty:
                    ax.scatter(anomalies[x_axis], anomalies[y_axis], color='red', label=f'이상치 ({len(anomalies)}건)', zorder=5)
                    
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.6)
                st.pyplot(fig)
                
            else:
                st.error("🚨 숫자 데이터가 포함된 열(Column)을 선택해주세요.")
        except Exception as e:
            st.error(f"파일을 읽을 수 없습니다. (에러: {e})")
