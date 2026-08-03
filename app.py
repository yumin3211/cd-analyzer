import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 웹사이트 기본 설정
st.set_page_config(page_title="Chirality Batch Analyzer", layout="wide")
st.title("📊 CD 스펙트럼 대량 분석 및 자동 진단 시스템")
st.write("다수의 실험 데이터를 한 번에 업로드하여 일괄 비교하고, 심층적인 결과 해석 레포트를 제공합니다.")

# 1. 파일 다중 업로드 (accept_multiple_files=True 옵션 추가)
col1, col2 = st.columns(2)
with col1:
    r_files = st.file_uploader("📂 R-form CSV 파일들 (드래그로 여러 개 선택 가능)", type=['csv'], accept_multiple_files=True)
with col2:
    s_files = st.file_uploader("📂 S-form CSV 파일들 (드래그로 여러 개 선택 가능)", type=['csv'], accept_multiple_files=True)

# 2. 파일이 업로드되면 일괄 분석 시작
if r_files and s_files:
    if len(r_files) != len(s_files):
        st.error("🚨 R-form과 S-form 파일의 개수가 다릅니다. 짝이 맞게 동일한 개수로 업로드해주세요.")
    else:
        # 파일 이름을 알파벳 순으로 정렬하여 R과 S의 짝을 자동으로 맞춥니다.
        r_files_sorted = sorted(r_files, key=lambda x: x.name)
        s_files_sorted = sorted(s_files, key=lambda x: x.name)
        
        summary_list = []
        detailed_reports = []

        # 데이터 일괄 처리 루프
        for r_file, s_file in zip(r_files_sorted, s_files_sorted):
            pair_name = f"{r_file.name} & {s_file.name}"
            
            # 전처리
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
                
            # 유효 구간(350nm 이하) 핵심 피크 찾기
            df_r_valid = df_r[df_r[0] <= 350]
            df_s_valid = df_s[df_s[0] <= 350]
            
            r_peak_wave = df_r[0].loc[df_r_valid[2].idxmax()]
            s_peak_wave = df_s[0].loc[df_s_valid[2].idxmin()]
            wave_diff = abs(r_peak_wave - s_peak_wave)
            
            # 대시보드용 요약 데이터 저장
            summary_list.append({
                "실험 데이터 그룹명": pair_name,
                "대칭성 점수(%)": round(similarity, 1),
                "R 최대 피크(nm)": r_peak_wave,
                "S 최소 피크(nm)": s_peak_wave,
                "파장 오차(nm)": round(wave_diff, 1)
            })
            
            # 상세 레포트용 데이터 통째로 저장
            detailed_reports.append({
                "name": pair_name,
                "similarity": similarity,
                "df_r": df_r, "df_s": df_s,
                "r_norm": r_norm, "s_norm": s_norm,
                "r_peak": r_peak_wave, "s_peak": s_peak_wave,
                "wave_diff": wave_diff
            })

        # ---------------------------------------------------------
        # 화면 출력 1: 전체 데이터 일괄 대시보드
        # ---------------------------------------------------------
        st.write("---")
        st.header("📋 전체 실험 그룹 일괄 평가 대시보드")
        st.write("여러 실험 데이터를 한눈에 비교하고 평가할 수 있는 통합 결과표입니다.")
        
        summary_df = pd.DataFrame(summary_list)
        # 점수가 높은 행에 시각적 하이라이트(색상)를 줍니다.
        st.dataframe(summary_df.style.background_gradient(subset=['대칭성 점수(%)'], cmap='Blues'), use_container_width=True)

        # ---------------------------------------------------------
        # 화면 출력 2: 개별 데이터 심층 해석 레포트 (아코디언 형태)
        # ---------------------------------------------------------
        st.write("---")
        st.header("🔎 개별 실험 결과 심층 해석 레포트")
        st.write("각 항목을 클릭하면 그래프와 함께 AI의 상세 피드백을 확인할 수 있습니다.")
        
        for report in detailed_reports:
            # st.expander를 사용해 클릭하면 열리는 보고서 양식을 만듭니다.
            with st.expander(f"📂 [{report['similarity']:.1f}%] {report['name']} 상세 분석 보기"):
                
                sim = report['similarity']
                diff = report['wave_diff']
                
                # 심층 해석 로직 3단계
                if sim >= 85:
                    eval_1 = "매우 우수함. 농도 보정 후 두 스펙트럼이 완벽한 거울상 대칭을 이루고 있으며, 광학적 순도(Enantiomeric Excess)가 높게 유지된 성공적인 결과입니다."
                    eval_3 = "어닐링(Annealing) 시간 및 용매 증발 속도가 완벽하게 통제되었습니다. 현재의 공정 조건을 표준 지침으로 채택할 수 있습니다."
                elif sim >= 65:
                    eval_1 = "보통 수준임. 거울상 경향성은 확인되나 부분적인 비대칭성이 존재합니다. 샘플 내 미세 불순물이나 농도 오차의 영향이 일부 남아있습니다."
                    eval_3 = "용매 증발 과정에서 미세한 환경 변인이 개입되었을 가능성이 있습니다. 추가적인 공정 최적화가 권장됩니다."
                else:
                    eval_1 = "경고 수준. 대칭성이 크게 훼손되었습니다. 물질 고유의 카이랄성이 제대로 발현되지 않았으며, 심각한 구조적 왜곡이 발생했습니다."
                    eval_3 = "용매 증발 속도의 불균형(Open 조건 등) 또는 어닐링 시간 부족으로 분자 배향이 무너진 것이 주요 원인으로 추정됩니다. 즉각적인 원인 규명이 필요합니다."
                    
                if diff <= 5:
                    eval_2 = f"R-form({report['r_peak']}nm)과 S-form({report['s_peak']}nm)의 핵심 피크 발생 위치가 일치하여, 구조적 동질성이 교차 검증되었습니다."
                else:
                    eval_2 = f"두 물질 간의 피크 파장 위치가 {diff:.1f}nm 어긋나 있습니다. 분자 간 상호작용의 차이나 측정 장비의 세팅 오류가 의심됩니다."

                st.info(f"**1. 구조 및 카이랄성 평가:** {eval_1}\n\n**2. 분광학적 피크 일치도:** {eval_2}\n\n**3. 종합 공정 피드백:** {eval_3}")
                
                plot_col1, plot_col2 = st.columns(2)
                with plot_col1:
                    st.caption("원본 데이터 (농도 오차 포함)")
                    fig1, ax1 = plt.subplots(figsize=(5, 3))
                    ax1.plot(report['df_r'][0], report['df_r'][2], label='R-form (Raw)', color='blue')
                    ax1.plot(report['df_s'][0], report['df_s'][2], label='S-form (Raw)', color='orange')
                    ax1.axhline(0, color='black', linewidth=0.5)
                    st.pyplot(fig1)

                with plot_col2:
                    st.caption("정규화 데이터 (보정 완료)")
                    fig2, ax2 = plt.subplots(figsize=(5, 3))
                    ax2.plot(report['df_r'][0], report['r_norm'], label='R-form (Norm)', color='blue')
                    ax2.plot(report['df_s'][0], report['s_norm'], label='S-form (Norm)', color='orange', linestyle='--')
                    ax2.axhline(0, color='black', linewidth=0.5)
                    st.pyplot(fig2)
