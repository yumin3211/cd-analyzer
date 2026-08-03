import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 웹사이트 기본 설정
st.set_page_config(page_title="Chirality Batch Analyzer", layout="wide")
st.title("📊 CD 스펙트럼 대량 분석 및 자동 진단 시스템")
st.write("실험 조건(Open/Half/Close)과 어닐링 시간이 포함된 파일명을 AI가 자동 인식하여 일괄 분석합니다.")

# 1. 파일 업로드 (R, S 구분 없이 한 번에 다 던져넣기)
st.info("💡 **[업로드 팁]** 파일 이름이 `R_open_30.csv`, `S_h_60.csv` 형식이어야 조건을 자동 인식합니다.")
uploaded_files = st.file_uploader("📂 실험 CSV 파일들을 모두 선택해서 올려주세요 (드래그 앤 드롭)", type=['csv'], accept_multiple_files=True)

if uploaded_files:
    experiments = {}
    
    # 2. 파일명 자동 분석 및 그룹화 (R과 S 짝맞추기)
    for f in uploaded_files:
        # 파일명에서 확장자 제거 및 '_' 기준으로 쪼개기 (예: R_open_30 -> ['R', 'open', '30'])
        clean_name = f.name.replace('.csv', '').replace('.CSV', '')
        parts = clean_name.split('_')
        
        if len(parts) >= 3:
            form = parts[0].upper()  # R 또는 S
            condition_raw = parts[1].lower() # open, h, close 등
            time_raw = parts[2] # 30, 60 등
            
            # 조건 이름 예쁘게 바꾸기
            if condition_raw == 'open': cond_display = "Open (공기 노출)"
            elif condition_raw in ['h', 'half']: cond_display = "Half-open (부분 차단)"
            elif condition_raw == 'close': cond_display = "Close (완전 차단)"
            else: cond_display = condition_raw
            
            exp_key = f"{cond_display}_{time_raw}분"
            
            # 딕셔너리에 방 만들기
            if exp_key not in experiments:
                experiments[exp_key] = {'R': None, 'S': None, 'Condition': cond_display, 'Time': f"{time_raw}분"}
                
            # R, S 파일 쏙쏙 집어넣기
            if form == 'R': experiments[exp_key]['R'] = f
            elif form == 'S': experiments[exp_key]['S'] = f

    # 3. 분석 시작
    summary_list = []
    detailed_reports = []

    for key, data in experiments.items():
        # R과 S 짝이 모두 있는 경우만 분석
        if data['R'] is not None and data['S'] is not None:
            r_file = data['R']
            s_file = data['S']
            cond = data['Condition']
            time = data['Time']
            
            # 데이터 읽기 및 전처리
            df_r = pd.read_csv(r_file, skiprows=21, header=None)
            df_s = pd.read_csv(s_file, skiprows=21, header=None)
            
            df_r[0] = pd.to_numeric(df_r[0], errors='coerce')
            df_r[2] = pd.to_numeric(df_r[2], errors='coerce')
            df_r = df_r.dropna(subset=[0, 2])
            
            df_s[0] = pd.to_numeric(df_s[0], errors='coerce')
            df_s[2] = pd.to_numeric(df_s[2], errors='coerce')
            df_s = df_s.dropna(subset=[0, 2])
            
            # 정규화
            r_norm = df_r[2] / df_r[2].abs().max()
            s_norm = df_s[2] / df_s[2].abs().max()
            
            # 대칭성 점수
            try:
                similarity = np.corrcoef(r_norm, -s_norm)[0, 1] * 100
            except:
                similarity = 0
                
            # 피크 분석 (350nm 이하 유효 구간)
            df_r_valid = df_r[df_r[0] <= 350]
            df_s_valid = df_s[df_s[0] <= 350]
            
            r_peak_wave = df_r[0].loc[df_r_valid[2].idxmax()]
            s_peak_wave = df_s[0].loc[df_s_valid[2].idxmin()]
            
            # 요약 데이터 저장
            summary_list.append({
                "실험 조건": cond,
                "어닐링 시간": time,
                "대칭성 점수(%)": round(similarity, 1),
                "R 최대 파장(nm)": r_peak_wave,
                "S 최소 파장(nm)": s_peak_wave,
            })
            
            # 상세 레포트 데이터 저장
            detailed_reports.append({
                "key": key, "cond": cond, "time": time,
                "similarity": similarity,
                "df_r": df_r, "df_s": df_s,
                "r_norm": r_norm, "s_norm": s_norm
            })

    # ---------------------------------------------------------
    # 결과 화면 출력
    # ---------------------------------------------------------
    if summary_list:
        st.write("---")
        st.header("📋 전체 실험 조건별 대시보드")
        
        summary_df = pd.DataFrame(summary_list)
        # 점수에 따라 색상 하이라이트 (점수가 높을수록 진한 초록색)
        st.dataframe(summary_df.style.background_gradient(subset=['대칭성 점수(%)'], cmap='Greens'), use_container_width=True)

        st.write("---")
        st.header("🔎 조건별 심층 해석 레포트")
        
        for report in detailed_reports:
            sim = report['similarity']
            title = f"📂 [{report['cond']} / {report['time']}] 대칭성: {sim:.1f}%"
            
            with st.expander(title):
                # 조건과 점수에 따른 맞춤형 피드백 생성
                if sim >= 85:
                    feedback = f"✅ **[{report['cond']}] 조건은 성공적입니다.** {report['time']}의 어닐링 시간이 고분자 자가조립에 충분했으며, 용매 증발 속도가 이상적으로 제어되어 완벽한 카이랄성을 띄고 있습니다."
                elif sim >= 65:
                    feedback = f"⚠️ **[{report['cond']}] 조건에 일부 오차가 있습니다.** {report['time']} 동안 증발은 일어났으나 미세한 불균형이 존재합니다. 어닐링 시간을 늘리거나 밀폐 조건을 조금 더 조정해 보세요."
                else:
                    feedback = f"🚨 **[{report['cond']}] 조건은 대칭성이 불량합니다.** {report['time']}의 어닐링으로는 배향이 무너졌을 확률이 높습니다. 너무 빠르거나(Open) 막힌(Close) 증발 속도가 원인으로 파악됩니다."
                
                st.info(feedback)
                
                plot_col1, plot_col2 = st.columns(2)
                with plot_col1:
                    st.caption("원본 데이터")
                    fig1, ax1 = plt.subplots(figsize=(5, 3))
                    ax1.plot(report['df_r'][0], report['df_r'][2], label='R-form', color='blue')
                    ax1.plot(report['df_s'][0], report['df_s'][2], label='S-form', color='orange')
                    ax1.axhline(0, color='black', linewidth=0.5)
                    ax1.legend()
                    st.pyplot(fig1)

                with plot_col2:
                    st.caption("정규화 데이터")
                    fig2, ax2 = plt.subplots(figsize=(5, 3))
                    ax2.plot(report['df_r'][0], report['r_norm'], label='R(Norm)', color='blue')
                    ax2.plot(report['df_s'][0], report['s_norm'], label='S(Norm)', color='orange', linestyle='--')
                    ax2.axhline(0, color='black', linewidth=0.5)
                    ax2.legend()
                    st.pyplot(fig2)
    else:
        st.warning("⚠️ 분석할 수 있는 R/S 파일 짝이 없거나 파일명 형식이 다릅니다. (예: R_open_30.csv, S_open_30.csv 형식으로 올려주세요)")
