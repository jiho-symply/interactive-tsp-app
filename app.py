import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import algorithms as algo
import time
import threading
import queue

# --- 1. 초기 설정 ---
st.set_page_config(
    page_title="TSP 시뮬레이터", 
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'n_cities' not in st.session_state: st.session_state.n_cities = 25
if 'cities' not in st.session_state:
    coords = np.round(np.random.rand(st.session_state.n_cities, 2) * 100, 1)
    st.session_state.cities = pd.DataFrame(coords, columns=['x', 'y'])
    st.session_state.paths = {k: [] for k in ["대학원생 최적화", "MILP Solver", "Nearest Neighbor", "k-opt", "Simulated Annealing"]}
    st.session_state.scores = {k: 0.0 for k in st.session_state.paths.keys()}
    st.session_state.times = {k: 0.0 for k in st.session_state.paths.keys()}

# --- 2. 그래프 함수 ---
def draw_tsp_plot(cities_df, path, title, color="orange"):
    n_cities = len(cities_df)
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=cities_df.x, y=cities_df.y,
        mode='markers+text', 
        text=[f"C{i+1}" for i in range(n_cities)],
        textposition="top center", 
        marker=dict(size=10, color='black'), 
        name="도시"
    ))
    
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
        xaxis=dict(visible=False, range=[-5, 105], constrain="domain", fixedrange=True),
        yaxis=dict(visible=False, range=[-5, 105], scaleanchor="x", scaleratio=1, fixedrange=True),
        height=900,
        showlegend=False,
        dragmode=False,
        title=f"{title} (거리: {algo.calculate_total_dist(path, cities_df)})"
    )
    return fig

chart_config = {'displayModeBar': False, 'scrollZoom': False}

# --- 3. 스레드 실행 도우미 (타이머 기능 추가) ---
def run_algorithm_in_background(target_func, args, graph_spot, chart_color, timer_spot=None):
    update_queue = queue.Queue()
    result_queue = queue.Queue()
    cities_copy = st.session_state.cities.copy()
    
    def thread_target():
        def callback_wrapper(p, t):
            update_queue.put((p, t))
        
        try:
            res = target_func(*args, callback=callback_wrapper)
            result_queue.put(res)
        except Exception as e:
            result_queue.put([])
            print(f"Error: {e}")

    t = threading.Thread(target=thread_target)
    t.start()
    
    start_time = time.time()
    update_idx = 0
    
    while t.is_alive():
        # [추가] 실시간 타이머 업데이트
        elapsed = time.time() - start_time
        if timer_spot:
            timer_spot.markdown(f"### ⏱️ 경과 시간: **{elapsed:.2f}s**")
        
        try:
            # 큐에서 데이터를 꺼내 그래프 업데이트
            path, title = update_queue.get(timeout=0.05)
            update_idx += 1
            graph_spot.plotly_chart(
                draw_tsp_plot(cities_copy, path, title, chart_color), 
                use_container_width=True, 
                config=chart_config,
                key=f"live_{chart_color}_{update_idx}"
            )
        except queue.Empty:
            pass
            
    t.join()
    end_time = time.time()
    
    # 최종 시간 표시 고정
    if timer_spot:
        timer_spot.markdown(f"### ⏱️ 완료 시간: **{end_time - start_time:.2f}s**")

    if not result_queue.empty():
        return result_queue.get(), end_time - start_time
    return [], 0.0

# --- 4. 사이드바 ---
with st.sidebar:
    st.header("🎮 맵 설정")
    num_cities = st.number_input("도시 개수 선택", min_value=5, max_value=100, value=st.session_state.n_cities)
    
    if st.button("도시 생성", use_container_width=True, type="primary"):
        st.session_state.n_cities = num_cities
        coords = np.round(np.random.rand(num_cities, 2) * 100, 1)
        st.session_state.cities = pd.DataFrame(coords, columns=['x', 'y'])
        st.session_state.paths = {k: [] for k in st.session_state.paths.keys()}
        st.session_state.scores = {k: 0.0 for k in st.session_state.paths.keys()}
        st.session_state.times = {k: 0.0 for k in st.session_state.paths.keys()}
        st.rerun()

# --- 5. 메인 화면 ---
st.title("🏙️ TSP 시뮬레이터")

st.subheader("📊 결과 비교표 (Leaderboard)")
res_data = []
best_dist = float('inf')

for k, path in st.session_state.paths.items():
    if path and len(path) == st.session_state.n_cities:
        d = st.session_state.scores[k]
        if d < best_dist: best_dist = d

for k, path in st.session_state.paths.items():
    dist = st.session_state.scores[k]
    exec_time = st.session_state.times.get(k, 0.0)
    
    if dist == 0: continue
    
    is_complete = len(path) == st.session_state.n_cities
    status_icon = "✅" if is_complete else "🚧"
    
    gap_str = "-"
    if is_complete and best_dist != float('inf'):
        if dist == best_dist: gap_str = "🏆 Best"
        else:
            diff = ((dist - best_dist) / best_dist) * 100
            gap_str = f"+{diff:.1f}%"
    
    res_data.append({
        "알고리즘": k, 
        "거리": dist, 
        "시간(s)": f"{exec_time:.2f}",
        "GAP": gap_str, 
        "상태": status_icon
    })

if res_data:
    df = pd.DataFrame(res_data).sort_values(by="거리").reset_index(drop=True)
    df.index += 1
    st.dataframe(
        df, 
        column_config={
            "알고리즘": st.column_config.TextColumn("알고리즘", width="medium"),
            "거리": st.column_config.NumberColumn("거리", format="%.1f"),
            "시간(s)": st.column_config.TextColumn("시간(s)"),
            "GAP": st.column_config.TextColumn("Gap"),
            "상태": st.column_config.TextColumn("완료")
        },
        use_container_width=True
    )
else:
    st.info("실행된 알고리즘이 없습니다.")

st.divider()

tabs = st.tabs(["✍️ 대학원생 최적화", "🏆 MILP Solver", "📍 Nearest Neighbor", "🔧 k-opt", "🔥 Simulated Annealing"])

# 1. 대학원생 최적화
with tabs[0]:
    st.info("💡 대학원생의 직관은 때론 휴리스틱보다 강력합니다. 점을 순서대로 클릭하여 경로를 설계하세요.")
    c1, c2 = st.columns([3, 1])
    if c2.button("🧹 경로 초기화", use_container_width=True):
        st.session_state.paths["대학원생 최적화"] = []
        st.session_state.scores["대학원생 최적화"] = 0.0
        st.session_state.times["대학원생 최적화"] = 0.0
        st.rerun()
        
    graph_spot = st.empty()
    selected = graph_spot.plotly_chart(
        draw_tsp_plot(st.session_state.cities, st.session_state.paths["대학원생 최적화"], "대학원생 최적화", "orange"), 
        on_select="rerun", use_container_width=True, config=chart_config
    )
    
    if selected and "selection" in selected and selected["selection"]["point_indices"]:
        idx = selected["selection"]["point_indices"][0]
        p = st.session_state.paths["대학원생 최적화"]
        if idx in p: p.remove(idx)
        else: p.append(idx)
        st.session_state.scores["대학원생 최적화"] = algo.calculate_total_dist(p, st.session_state.cities)
        st.rerun()

# 2. MILP Solver (Optimal)
with tabs[1]:
    st.markdown("> **MILP Solver**: 수학적 모델링(CP-SAT)을 통해 증명된 전역 최적해(Global Optimum)를 도출합니다.")
    
    c1, c2 = st.columns([3, 1])
    # [수정] 기본 시간 10초
    timeout = c1.slider("실행 시간 제한 (초)", 1, 60, 10, key="milp_time")
    timer_spot = c1.empty() # 타이머 표시 위치
    
    graph_spot = st.empty()
    if c2.button("알고리즘 실행", key="opt", type="primary", use_container_width=True):
        res, t = run_algorithm_in_background(
            algo.run_optimal_solver, 
            (st.session_state.cities, timeout), 
            graph_spot, "gold", timer_spot
        )
        st.session_state.paths["MILP Solver"] = res
        st.session_state.scores["MILP Solver"] = algo.calculate_total_dist(res, st.session_state.cities)
        st.session_state.times["MILP Solver"] = t
        st.rerun()
    else: 
        graph_spot.plotly_chart(draw_tsp_plot(st.session_state.cities, st.session_state.paths["MILP Solver"], "MILP 최적해", "gold"), use_container_width=True, config=chart_config)

# 3. Nearest Neighbor
with tabs[2]:
    st.markdown("> **Nearest Neighbor**: 현재 위치에서 가장 가까운 도시를 찾아가는 탐욕 알고리즘입니다.")
    c1, c2 = st.columns([3, 1])
    start_node = c1.selectbox("시작 도시", range(st.session_state.n_cities), format_func=lambda x: f"도시 {x+1}")
    timer_spot = c1.empty()
    
    graph_spot = st.empty()
    if c2.button("알고리즘 실행", key="nn", type="primary", use_container_width=True):
        res, t = run_algorithm_in_background(
            algo.run_nn, 
            (st.session_state.n_cities, start_node, st.session_state.cities), 
            graph_spot, "royalblue", timer_spot
        )
        st.session_state.paths["Nearest Neighbor"] = res
        st.session_state.scores["Nearest Neighbor"] = algo.calculate_total_dist(res, st.session_state.cities)
        st.session_state.times["Nearest Neighbor"] = t
        st.rerun()
    else: 
        graph_spot.plotly_chart(draw_tsp_plot(st.session_state.cities, st.session_state.paths["Nearest Neighbor"], "NN 결과", "royalblue"), use_container_width=True, config=chart_config)

# 4. k-opt
with tabs[3]:
    st.markdown("> **k-opt**: 경로의 일부를 끊고 재연결하여 거리를 줄이는 지역 탐색 알고리즘입니다.")
    c1, c2 = st.columns([3, 1])
    k_v = c1.radio("방식 선택", ["2-opt", "3-opt"], horizontal=True)
    # [수정] 기본 시간 10초
    timeout = c1.slider("실행 시간 제한 (초)", 1, 60, 10, key="kopt_time")
    timer_spot = c1.empty()
    
    graph_spot = st.empty()
    if c2.button("알고리즘 실행", key="kopt", type="primary", use_container_width=True):
        res, t = run_algorithm_in_background(
            algo.run_kopt, 
            (k_v, st.session_state.cities, timeout), 
            graph_spot, "green", timer_spot
        )
        st.session_state.paths["k-opt"] = res
        st.session_state.scores["k-opt"] = algo.calculate_total_dist(res, st.session_state.cities)
        st.session_state.times["k-opt"] = t
        st.rerun()
    else: 
        graph_spot.plotly_chart(draw_tsp_plot(st.session_state.cities, st.session_state.paths["k-opt"], "k-opt 결과", "green"), use_container_width=True, config=chart_config)

# 5. Simulated Annealing
with tabs[4]:
    # [수정] 이웃해 탐색 방법(Neighborhood Search) 명시
    st.markdown("""
    > **Simulated Annealing**: 확률적 메타휴리스틱으로, 초기 온도가 높을수록 나쁜 해를 더 잘 수용합니다.
    > * **Neighbor Search Operators**: OR-Tools가 내부적으로 **Relocate, Exchange, Cross, 2-opt** 등의 연산자를 조합하여 이웃해를 탐색합니다.
    """)
    
    c1, c2 = st.columns([3, 1])
    
    # [수정] SA 상세 옵션 추가
    with c1:
        c1_1, c1_2 = st.columns(2)
        init_strategy = c1_1.selectbox(
            "초기 해 생성 (Initialization)", 
            ["Automatic (Default)", "Greedy (Path Cheapest)", "Savings", "Christofides", "Random"],
            index=0
        )
        init_temp = c1_2.number_input("초기 온도 (Initial Temp)", min_value=0, value=100, help="0이면 자동 설정")
        # [수정] 기본 시간 10초
        timeout = st.slider("실행 시간 제한 (초)", 1, 60, 10, key="sa_time")
        timer_spot = st.empty()
    
    graph_spot = st.empty()
    if c2.button("알고리즘 실행", key="sa", type="primary", use_container_width=True):
        # 0이면 None으로 전달하여 OR-Tools 자동 설정 사용
        temp_arg = init_temp if init_temp > 0 else None
        
        res, t = run_algorithm_in_background(
            algo.run_sa, 
            (st.session_state.cities, timeout, init_strategy, temp_arg), 
            graph_spot, "purple", timer_spot
        )
        st.session_state.paths["Simulated Annealing"] = res
        st.session_state.scores["Simulated Annealing"] = algo.calculate_total_dist(res, st.session_state.cities)
        st.session_state.times["Simulated Annealing"] = t
        st.rerun()
    else: 
        graph_spot.plotly_chart(draw_tsp_plot(st.session_state.cities, st.session_state.paths["Simulated Annealing"], "SA 결과", "purple"), use_container_width=True, config=chart_config)
