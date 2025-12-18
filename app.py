import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import algorithms as algo
import time

# --- 1. 초기 설정 및 세션 관리 ---
st.set_page_config(page_title="TSP 시뮬레이터", layout="wide")
st.title("🏙️ TSP 시뮬레이터 (C++ Engine Edition)")

if 'n_cities' not in st.session_state: st.session_state.n_cities = 25
if 'cities' not in st.session_state:
    coords = np.round(np.random.rand(st.session_state.n_cities, 2) * 100, 1)
    st.session_state.cities = pd.DataFrame(coords, columns=['x', 'y'])
    st.session_state.paths = {k: [] for k in ["대학원생 최적화", "Nearest Neighbor", "k-opt", "Simulated Annealing", "Advanced (GLS)"]}
    st.session_state.scores = {k: 0.0 for k in st.session_state.paths.keys()}

@st.dialog("새 도시 배치")
def reset_cities_dialog():
    st.write("도시 개수(5~50)를 선택하세요.")
    num = st.number_input("도시 개수", 5, 50, st.session_state.n_cities)
    c1, c2 = st.columns(2)
    if c1.button("취소", use_container_width=True): st.rerun()
    if c2.button("배치 생성", use_container_width=True, type="primary"):
        st.session_state.n_cities = num
        coords = np.round(np.random.rand(num, 2) * 100, 1)
        st.session_state.cities = pd.DataFrame(coords, columns=['x', 'y'])
        st.session_state.paths = {k: [] for k in st.session_state.paths.keys()}
        st.session_state.scores = {k: 0.0 for k in st.session_state.paths.keys()}
        st.rerun()

# --- 2. 그래프 렌더링 함수 (Interaction 고정) ---
def draw_tsp_plot(path, title, color="orange"):
    fig = go.Figure()
    # 도시 포인트 (1부터 시작하는 라벨)
    fig.add_trace(go.Scatter(
        x=st.session_state.cities.x, y=st.session_state.cities.y,
        mode='markers+text', 
        text=[f"C{i+1}" for i in range(st.session_state.n_cities)],
        textposition="top center", 
        marker=dict(size=10, color='black'), 
        name="도시"
    ))
    # 경로 선
    if path and len(path) > 0:
        d_path = path + [path[0]] if len(path) == st.session_state.n_cities else path
        coords = st.session_state.cities.iloc[d_path]
        fig.add_trace(go.Scatter(
            x=coords.x, y=coords.y, 
            mode='lines+markers', 
            line=dict(color=color, width=3),
            hoverinfo="skip"
        ))
    
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(showgrid=False, range=[-5, 105], constrain="domain", fixedrange=True),
        yaxis=dict(showgrid=False, range=[-5, 105], scaleanchor="x", scaleratio=1, fixedrange=True),
        height=900,
        showlegend=False,
        dragmode=False,
        title=f"{title} (거리: {algo.calculate_total_dist(path, st.session_state.cities)})"
    )
    return fig

chart_config = {'displayModeBar': False, 'scrollZoom': False}

# --- 3. 메인 레이아웃 ---
col_main, col_side = st.columns([3, 1])

with col_side:
    st.subheader("📊 결과 비교표")
    score_list = [{"알고리즘": k, "거리": v} for k, v in st.session_state.scores.items() if v > 0]
    if score_list:
        df = pd.DataFrame(score_list).sort_values(by="거리").reset_index(drop=True)
        df.index += 1
        df.index.name = "순위"
        st.table(df.style.format({"거리": "{:.1f}"}))
    else: st.info("데이터 없음")
    if st.button("🗺️ 새 도시 배치", use_container_width=True): reset_cities_dialog()

with col_main:
    tabs = st.tabs(["✍️ 대학원생 최적화", "📍 Nearest Neighbor", "🔧 k-opt", "🔥 Simulated Annealing", "🚀 Advanced (GLS)"])

    # --- 탭 1: 대학원생 최적화 (수동) ---
    with tabs[0]:
        st.info("💡 대학원생의 직관 모드: 점을 순서대로 클릭하세요. 다시 누르면 취소됩니다.")
        c1, c2 = st.columns([3, 1])
        if c2.button("🧹 경로 초기화", use_container_width=True):
            st.session_state.paths["대학원생 최적화"] = []
            st.session_state.scores["대학원생 최적화"] = 0.0
            st.rerun()
        graph_spot = st.empty()
        selected = graph_spot.plotly_chart(draw_tsp_plot(st.session_state.paths["대학원생 최적화"], "대학원생 최적화", "orange"), on_select="rerun", use_container_width=True, config=chart_config)
        if selected and "selection" in selected and selected["selection"]["point_indices"]:
            idx = selected["selection"]["point_indices"][0]
            p = st.session_state.paths["대학원생 최적화"]
            if idx in p: p.remove(idx)
            else: p.append(idx)
            st.session_state.scores["대학원생 최적화"] = algo.calculate_total_dist(p, st.session_state.cities)
            st.rerun()

    # --- 탭 2: Nearest Neighbor (OR-Tools 기반) ---
    with tabs[1]:
        st.markdown("> **C++ 기반 Nearest Neighbor**: OR-Tools의 Path Cheapest Arc 전략을 사용합니다.")
        c1, c2 = st.columns([3, 1])
        start_node = c1.selectbox("시작 도시 선택", range(st.session_state.n_cities), format_func=lambda x: f"도시 {x+1}")
        graph_spot = st.empty()
        if c2.button("알고리즘 실행", key="nn_run", type="primary", use_container_width=True):
            def cb(p, t): graph_spot.plotly_chart(draw_tsp_plot(p, t, "royalblue"), use_container_width=True, config=chart_config)
            res = algo.run_nn_engine(st.session_state.n_cities, start_node, st.session_state.cities, cb)
            st.session_state.paths["Nearest Neighbor"] = res
            st.session_state.scores["Nearest Neighbor"] = algo.calculate_total_dist(res, st.session_state.cities)
            st.rerun()
        else: graph_spot.plotly_chart(draw_tsp_plot(st.session_state.paths["Nearest Neighbor"], "NN 결과", "royalblue"), use_container_width=True, config=chart_config)

    # --- 탭 3: k-opt (C++ Local Search) ---
    with tabs[2]:
        st.markdown("> **C++ 기반 k-opt**: OR-Tools 내장 Local Search 연산자가 개선이 없을 때까지 수행합니다.")
        c1, c2 = st.columns([3, 1])
        k_v = c1.radio("알고리즘 선택", ["2-opt", "3-opt"], horizontal=True)
        graph_spot = st.empty()
        if c2.button("알고리즘 실행", key="kopt_run", type="primary", use_container_width=True):
            def cb(p, t): graph_spot.plotly_chart(draw_tsp_plot(p, t, "green"), use_container_width=True, config=chart_config)
            res = algo.run_kopt_engine(k_v, st.session_state.cities, cb)
            st.session_state.paths["k-opt"] = res
            st.session_state.scores["k-opt"] = algo.calculate_total_dist(res, st.session_state.cities)
            st.rerun()
        else: graph_spot.plotly_chart(draw_tsp_plot(st.session_state.paths["k-opt"], "k-opt 결과", "green"), use_container_width=True, config=chart_config)

    # --- 탭 4: Simulated Annealing (C++ Metaheuristic) ---
    with tabs[3]:
        st.markdown("> **C++ 기반 SA**: OR-Tools의 Simulated Annealing 메타휴리스틱 엔진을 사용합니다.")
        c1, c2 = st.columns([3, 1])
        graph_spot = st.empty()
        if c2.button("알고리즘 실행", key="sa_run", type="primary", use_container_width=True):
            def cb(p, t): graph_spot.plotly_chart(draw_tsp_plot(p, t, "purple"), use_container_width=True, config=chart_config)
            res = algo.run_sa_engine(st.session_state.cities, cb)
            st.session_state.paths["Simulated Annealing"] = res
            st.session_state.scores["Simulated Annealing"] = algo.calculate_total_dist(res, st.session_state.cities)
            st.rerun()
        else: graph_spot.plotly_chart(draw_tsp_plot(st.session_state.paths["Simulated Annealing"], "SA 결과", "purple"), use_container_width=True, config=chart_config)

    # --- 탭 5: Advanced (Guided Local Search) ---
    with tabs[4]:
        st.markdown("> **Advanced Optimal**: OR-Tools의 가장 강력한 전략인 Guided Local Search를 수행합니다.")
        c1, c2 = st.columns([3, 1])
        graph_spot = st.empty()
        if c2.button("알고리즘 실행", key="adv_run", type="primary", use_container_width=True):
            def cb(p, t): graph_spot.plotly_chart(draw_tsp_plot(p, t, "gold"), use_container_width=True, config=chart_config)
            res = algo.run_advanced_engine(st.session_state.cities, cb)
            st.session_state.paths["Advanced (GLS)"] = res
            st.session_state.scores["Advanced (GLS)"] = algo.calculate_total_dist(res, st.session_state.cities)
            st.rerun()
        else: graph_spot.plotly_chart(draw_tsp_plot(st.session_state.paths["Advanced (GLS)"], "GLS 결과", "gold"), use_container_width=True, config=chart_config)
