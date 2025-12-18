import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import algorithms as algo
import time

# --- 1. 초기 설정 (사이드바 확장) ---
st.set_page_config(
    page_title="TSP 시뮬레이터", 
    layout="wide",
    initial_sidebar_state="expanded"  # [수정] 사이드바 기본 열림
)

if 'n_cities' not in st.session_state: st.session_state.n_cities = 25
if 'cities' not in st.session_state:
    coords = np.round(np.random.rand(st.session_state.n_cities, 2) * 100, 1)
    st.session_state.cities = pd.DataFrame(coords, columns=['x', 'y'])
    # 탭 순서 변경에 따른 키 순서
    st.session_state.paths = {k: [] for k in ["대학원생 최적화", "MILP Solver", "Nearest Neighbor", "k-opt", "Simulated Annealing"]}
    st.session_state.scores = {k: 0.0 for k in st.session_state.paths.keys()}

# --- 2. 그래프 렌더링 함수 (에러 수정: 데이터 인자 전달) ---
def draw_tsp_plot(cities_df, path, title, color="orange"):
    # [수정] cities_df를 인자로 받아 세션 상태 의존성 제거
    n_cities = len(cities_df)
    fig = go.Figure()
    
    # 도시 그리기
    fig.add_trace(go.Scatter(
        x=cities_df.x, y=cities_df.y,
        mode='markers+text', 
        text=[f"C{i+1}" for i in range(n_cities)],
        textposition="top center", 
        marker=dict(size=10, color='black'), 
        name="도시"
    ))
    
    # 경로 그리기
    if path and len(path) > 0:
        d_path = path + [path[0]] if len(path) == n_cities else path
        coords = cities_df.iloc[d_path]
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
        height=900,  # [수정] 높이 900으로 복구
        showlegend=False,
        dragmode=False,
        title=f"{title} (거리: {algo.calculate_total_dist(path, cities_df)})"
    )
    return fig

chart_config = {'displayModeBar': False, 'scrollZoom': False}

# --- 3. 사이드바 ---
with st.sidebar:
    st.header("🎮 컨트롤 패널")
    st.subheader("맵 설정")
    num_cities = st.number_input("도시 개수 선택", min_value=5, max_value=50, value=st.session_state.n_cities)
    
    if st.button("도시 생성", use_container_width=True, type="primary"):
        st.session_state.n_cities = num_cities
        coords = np.round(np.random.rand(num_cities, 2) * 100, 1)
        st.session_state.cities = pd.DataFrame(coords, columns=['x', 'y'])
        st.session_state.paths = {k: [] for k in st.session_state.paths.keys()}
        st.session_state.scores = {k: 0.0 for k in st.session_state.paths.keys()}
        st.rerun()
    
    st.divider()
    st.subheader("📊 결과 비교 (Leaderboard)")
    
    res_data = []
    best_dist = float('inf')
    
    # 최적값 탐색
    for k, path in st.session_state.paths.items():
        if path and len(path) == st.session_state.n_cities:
            d = st.session_state.scores[k]
            if d < best_dist: best_dist = d
            
    # 데이터 생성
    for k, path in st.session_state.paths.items():
        dist = st.session_state.scores[k]
        if dist == 0: continue
        
        is_complete = len(path) == st.session_state.n_cities
        status_icon = "✅" if is_complete else "🚧"
        
        gap_str = "-"
        if is_complete and best_dist != float('inf'):
            if dist == best_dist: gap_str = "🏆 Best"
            else:
                diff = ((dist - best_dist) / best_dist) * 100
                gap_str = f"+{diff:.1f}%"
        
        res_data.append({"알고리즘": k, "거리": dist, "GAP": gap_str, "상태": status_icon})
    
    if res_data:
        df = pd.DataFrame(res_data).sort_values(by="거리").reset_index(drop=True)
        df.index += 1
        st.dataframe(
            df, 
            column_config={
                "알고리즘": st.column_config.TextColumn("알고리즘", width="medium"),
                "거리": st.column_config.NumberColumn("거리", format="%.1f"),
                "GAP": st.column_config.TextColumn("Gap"),
                "상태": st.column_config.TextColumn("완료")
            },
            use_container_width=True
        )
    else:
        st.info("실행된 알고리즘이 없습니다.")

# --- 4. 메인 탭 화면 ---
st.title("🏙️ TSP 시뮬레이터")

# [수정] 탭 순서 재배치
tabs = st.tabs(["✍️ 대학원생 최적화", "🏆 MILP Solver", "📍 Nearest Neighbor", "🔧 k-opt", "🔥 Simulated Annealing"])

# 1. 대학원생 최적화
with tabs[0]:
    st.info("💡 대학원생의 직관은 때론 휴리스틱보다 강력합니다. 점을 순서대로 클릭하여 경로를 설계하세요.")
    c1, c2 = st.columns([3, 1])
    if c2.button("🧹 경로 초기화", use_container_width=True):
        st.session_state.paths["대학원생 최적화"] = []
        st.session_state.scores["대학원생 최적화"] = 0.0
        st.rerun()
        
    graph_spot = st.empty()
    # draw_tsp_plot 호출 시 st.session_state.cities 전달
    selected = graph_spot.plotly_chart(
        draw_tsp_plot(st.session_state.cities, st.session_state.paths["대학원생 최적화"], "대학원생 최적화", "orange"), 
        on_select="rerun", 
        use_container_width=True, 
        config=chart_config
    )
    
    if selected and "selection" in selected and selected["selection"]["point_indices"]:
        idx = selected["selection"]["point_indices"][0]
        p = st.session_state.paths["대학원생 최적화"]
        if idx in p: p.remove(idx)
        else: p.append(idx)
        st.session_state.scores["대학원생 최적화"] = algo.calculate_total_dist(p, st.session_state.cities)
        st.rerun()

# 2. MILP Solver (Optimal) - [수정] 순서 이동 및 에러 해결
with tabs[1]:
    st.markdown("> **MILP Solver**: 수학적 모델링(CP-SAT)을 통해 증명된 전역 최적해(Global Optimum)를 도출합니다.")
    
    c1, c2 = st.columns([3, 1])
    graph_spot = st.empty()
    if c2.button("알고리즘 실행", key="opt", type="primary", use_container_width=True):
        # 콜백 함수: cities 데이터프레임을 클로저(closure)로 전달하여 안전하게 사용
        cities_copy = st.session_state.cities.copy()
        def cb(p, t): 
            graph_spot.plotly_chart(draw_tsp_plot(cities_copy, p, t, "gold"), use_container_width=True, config=chart_config)
            
        res = algo.run_optimal_solver(st.session_state.cities, cb)
        st.session_state.paths["MILP Solver"] = res
        st.session_state.scores["MILP Solver"] = algo.calculate_total_dist(res, st.session_state.cities)
        st.rerun()
    else: 
        graph_spot.plotly_chart(draw_tsp_plot(st.session_state.cities, st.session_state.paths["MILP Solver"], "MILP 최적해", "gold"), use_container_width=True, config=chart_config)

# 3. Nearest Neighbor
with tabs[2]:
    st.markdown("> **Nearest Neighbor**: 현재 위치에서 가장 가까운 도시를 찾아가는 탐욕 알고리즘입니다.")
    c1, c2 = st.columns([3, 1])
    start_node = c1.selectbox("시작 도시", range(st.session_state.n_cities), format_func=lambda x: f"도시 {x+1}")
    graph_spot = st.empty()
    if c2.button("알고리즘 실행", key="nn", type="primary", use_container_width=True):
        cities_copy = st.session_state.cities.copy()
        def cb(p, t): graph_spot.plotly_chart(draw_tsp_plot(cities_copy, p, t, "royalblue"), use_container_width=True, config=chart_config)
        res = algo.run_nn(st.session_state.n_cities, start_node, st.session_state.cities, cb)
        st.session_state.paths["Nearest Neighbor"] = res; st.session_state.scores["Nearest Neighbor"] = algo.calculate_total_dist(res, st.session_state.cities); st.rerun()
    else: graph_spot.plotly_chart(draw_tsp_plot(st.session_state.cities, st.session_state.paths["Nearest Neighbor"], "NN 결과", "royalblue"), use_container_width=True, config=chart_config)

# 4. k-opt
with tabs[3]:
    st.markdown("> **k-opt**: 경로의 일부를 끊고 재연결하여 거리를 줄이는 지역 탐색 알고리즘입니다.")
    c1, c2 = st.columns([3, 1])
    k_v = c1.radio("방식 선택", ["2-opt", "3-opt"], horizontal=True)
    graph_spot = st.empty()
    if c2.button("알고리즘 실행", key="kopt", type="primary", use_container_width=True):
        cities_copy = st.session_state.cities.copy()
        def cb(p, t): graph_spot.plotly_chart(draw_tsp_plot(cities_copy, p, t, "green"), use_container_width=True, config=chart_config)
        res = algo.run_kopt(k_v, st.session_state.cities, cb)
        st.session_state.paths["k-opt"] = res; st.session_state.scores["k-opt"] = algo.calculate_total_dist(res, st.session_state.cities); st.rerun()
    else: graph_spot.plotly_chart(draw_tsp_plot(st.session_state.cities, st.session_state.paths["k-opt"], "k-opt 결과", "green"), use_container_width=True, config=chart_config)

# 5. Simulated Annealing
with tabs[4]:
    st.markdown("> **Simulated Annealing**: 확률적으로 나쁜 해를 수용하며 전역 최적해를 찾는 담금질 기법입니다.")
    c1, c2 = st.columns([3, 1])
    graph_spot = st.empty()
    if c2.button("알고리즘 실행", key="sa", type="primary", use_container_width=True):
        cities_copy = st.session_state.cities.copy()
        def cb(p, t): graph_spot.plotly_chart(draw_tsp_plot(cities_copy, p, t, "purple"), use_container_width=True, config=chart_config)
        res = algo.run_sa(st.session_state.cities, cb)
        st.session_state.paths["Simulated Annealing"] = res; st.session_state.scores["Simulated Annealing"] = algo.calculate_total_dist(res, st.session_state.cities); st.rerun()
    else: graph_spot.plotly_chart(draw_tsp_plot(st.session_state.cities, st.session_state.paths["Simulated Annealing"], "SA 결과", "purple"), use_container_width=True, config=chart_config)
