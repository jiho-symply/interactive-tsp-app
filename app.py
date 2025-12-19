import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import algorithms as algo
import time
import threading
import queue
import traceback
import hashlib
import os

# --- 1. 초기 설정 ---
st.set_page_config(page_title="TSP 시뮬레이터", layout="wide", initial_sidebar_state="expanded")

# 기본 알고리즘 키 (고정적으로 표시할 것들)
FIXED_ALGORITHMS = ["대학원생 최적화", "MILP Solver", "Nearest Neighbor", "k-opt", "Metaheuristic"]

if 'n_cities' not in st.session_state: st.session_state.n_cities = 25
if 'cities' not in st.session_state:
    coords = np.round(np.random.rand(st.session_state.n_cities, 2) * 100, 1)
    st.session_state.cities = pd.DataFrame(coords, columns=['x', 'y'])
    # 경로 저장소: 기본 알고리즘 + 동적으로 추가될 Neural 모델들
    st.session_state.paths = {k: [] for k in FIXED_ALGORITHMS}
    st.session_state.scores = {k: 0.0 for k in FIXED_ALGORITHMS}
    st.session_state.times = {k: 0.0 for k in FIXED_ALGORITHMS}

# --- 2. 그래프 함수 ---
def draw_tsp_plot(cities_df, path, title, color="orange"):
    n_cities = len(cities_df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cities_df.x, y=cities_df.y, mode='markers+text', text=[f"C{i+1}" for i in range(n_cities)], textposition="top center", marker=dict(size=10, color='black'), name="도시"))
    if path and len(path) > 0:
        d_path = path + [path[0]] if len(path) == n_cities else path
        coords = cities_df.iloc[d_path]
        fig.add_trace(go.Scatter(x=coords.x, y=coords.y, mode='lines+markers', line=dict(color=color, width=3), hoverinfo="skip"))
    fig.update_layout(template="plotly_white", xaxis=dict(visible=False, range=[-5, 105], constrain="domain", fixedrange=True), yaxis=dict(visible=False, range=[-5, 105], scaleanchor="x", scaleratio=1, fixedrange=True), height=900, showlegend=False, dragmode=False, title=f"{title} (거리: {algo.calculate_total_dist(path, cities_df)})")
    return fig
chart_config = {'displayModeBar': False, 'scrollZoom': False}

# --- 3. 스레드 실행 도우미 ---
def run_algorithm_in_background(target_func, args, graph_spot, chart_color, timer_spot=None):
    update_queue = queue.Queue()
    result_queue = queue.Queue()
    cities_copy = st.session_state.cities.copy()
    
    def thread_target():
        def callback_wrapper(p, t): update_queue.put((p, t))
        try:
            res = target_func(*args, callback=callback_wrapper)
            result_queue.put(res)
        except Exception as e:
            err_msg = f"ERROR: {str(e)}\n{traceback.format_exc()}"
            result_queue.put(err_msg)

    t = threading.Thread(target=thread_target)
    t.start()
    start_time = time.time()
    update_idx = 0
    
    while t.is_alive():
        elapsed = time.time() - start_time
        if timer_spot: timer_spot.markdown(f"### ⏱️ 경과 시간: **{elapsed:.2f}s**")
        try:
            path, title = update_queue.get(timeout=0.01)
            update_idx += 1
            graph_spot.plotly_chart(draw_tsp_plot(cities_copy, path, title, chart_color), config=chart_config, key=f"live_{chart_color}_{update_idx}")
        except queue.Empty: pass
            
    t.join()
    end_time = time.time()
    if timer_spot: timer_spot.markdown(f"### ⏱️ 완료 시간: **{end_time - start_time:.2f}s**")

    if not result_queue.empty():
        res = result_queue.get()
        if isinstance(res, str) and res.startswith("ERROR"):
            st.error("알고리즘 실행 중 오류가 발생했습니다.")
            st.code(res)
            return [], 0.0
        return res, end_time - start_time
    return [], 0.0

# --- 4. 사이드바 ---
with st.sidebar:
    st.header("🎮 맵 설정")
    num_cities = st.number_input("도시 개수 선택", min_value=5, max_value=100, value=st.session_state.n_cities)
    seed_text = st.text_input("도시 생성 코드", placeholder="예: map1 (비워두면 랜덤)", help="특정 코드를 입력하면 항상 동일한 위치에 도시가 생성됩니다.")
    
    if st.button("도시 생성", use_container_width=True, type="primary"):
        st.session_state.n_cities = num_cities
        if seed_text:
            seed_val = int(hashlib.md5(seed_text.encode('utf-8')).hexdigest(), 16) % (2**32)
            np.random.seed(seed_val)
        coords = np.round(np.random.rand(num_cities, 2) * 100, 1)
        if seed_text: np.random.seed(None)
            
        st.session_state.cities = pd.DataFrame(coords, columns=['x', 'y'])
        # [중요] 기존 키는 유지하되 값만 초기화 (Neural 모델 기록 유지 여부는 선택 가능, 여기선 초기화)
        st.session_state.paths = {k: [] for k in st.session_state.paths.keys()} 
        st.session_state.scores = {k: 0.0 for k in st.session_state.paths.keys()}
        st.session_state.times = {k: 0.0 for k in st.session_state.paths.keys()}
        st.rerun()

# --- 5. 메인 화면 ---
st.title("🏙️ TSP 시뮬레이터")

st.subheader("📊 결과 비교표 (Leaderboard)")
res_data = []
best_dist = float('inf')

# 최단 거리 계산 (완료된 경로 중)
for k, path in st.session_state.paths.items():
    if path and len(path) == st.session_state.n_cities:
        d = st.session_state.scores[k]
        if d > 0 and d < best_dist: best_dist = d

# 리더보드 데이터 생성 (동적 키 지원)
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
    
    # Neural 모델인 경우 접두어 처리 (원한다면)
    display_name = k
    
    res_data.append({
        "알고리즘": display_name, 
        "거리": dist, 
        "시간(s)": f"{exec_time:.2f}",
        "GAP": gap_str, 
        "상태": status_icon
    })

if res_data:
    df = pd.DataFrame(res_data).sort_values(by="거리").reset_index(drop=True)
    df.index += 1
    st.dataframe(df, column_config={"알고리즘": st.column_config.TextColumn("알고리즘", width="medium"), "거리": st.column_config.NumberColumn("거리", format="%.1f"), "시간(s)": st.column_config.TextColumn("시간(s)"), "GAP": st.column_config.TextColumn("Gap"), "상태": st.column_config.TextColumn("완료")}, width="stretch")
else:
    st.info("실행된 알고리즘이 없습니다.")

st.divider()

tabs = st.tabs(["✍️ 대학원생 최적화", "🏆 MILP Solver", "📍 Nearest Neighbor", "🔧 k-opt", "🧩 Metaheuristic", "🧠 Neural Network"])

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
    selected = graph_spot.plotly_chart(draw_tsp_plot(st.session_state.cities, st.session_state.paths["대학원생 최적화"], "대학원생 최적화", "orange"), on_select="rerun", config=chart_config)
    if selected and "selection" in selected and selected["selection"]["point_indices"]:
        idx = selected["selection"]["point_indices"][0]
        p = st.session_state.paths["대학원생 최적화"]
        if idx in p: p.remove(idx)
        else: p.append(idx)
        st.session_state.scores["대학원생 최적화"] = algo.calculate_total_dist(p, st.session_state.cities)
        st.rerun()

# 2. MILP Solver
with tabs[1]:
    st.markdown("> **MILP Solver**: 수학적 모델링(CP-SAT)을 통해 증명된 전역 최적해를 도출합니다.")
    
    c1, c2 = st.columns([3, 1])
    timeout = c1.slider("실행 시간 제한 (초)", 1, 20, 5, key="milp_time")
    timer_spot = c1.empty()
    graph_spot = st.empty()
    if c2.button("알고리즘 실행", key="opt", type="primary", use_container_width=True):
        res, t = run_algorithm_in_background(algo.run_optimal_solver, (st.session_state.cities, timeout), graph_spot, "gold", timer_spot)
        if res:
            st.session_state.paths["MILP Solver"] = res
            st.session_state.scores["MILP Solver"] = algo.calculate_total_dist(res, st.session_state.cities)
            st.session_state.times["MILP Solver"] = t
            st.rerun()
    else: graph_spot.plotly_chart(draw_tsp_plot(st.session_state.cities, st.session_state.paths["MILP Solver"], "MILP 최적해", "gold"), config=chart_config)

# 3. Nearest Neighbor
with tabs[2]:
    st.markdown("> **Nearest Neighbor**: 현재 위치에서 가장 가까운 도시를 찾아가는 탐욕 알고리즘입니다.")
    c1, c2 = st.columns([3, 1])
    start_node = c1.selectbox("시작 도시", range(st.session_state.n_cities), format_func=lambda x: f"도시 {x+1}")
    timer_spot = c1.empty()
    graph_spot = st.empty()
    if c2.button("알고리즘 실행", key="nn", type="primary", use_container_width=True):
        res, t = run_algorithm_in_background(algo.run_nn, (st.session_state.n_cities, start_node, st.session_state.cities), graph_spot, "royalblue", timer_spot)
        if res:
            st.session_state.paths["Nearest Neighbor"] = res
            st.session_state.scores["Nearest Neighbor"] = algo.calculate_total_dist(res, st.session_state.cities)
            st.session_state.times["Nearest Neighbor"] = t
            st.rerun()
    else: graph_spot.plotly_chart(draw_tsp_plot(st.session_state.cities, st.session_state.paths["Nearest Neighbor"], "NN 결과", "royalblue"), config=chart_config)

# 4. k-opt
with tabs[3]:
    st.markdown("> **k-opt**: 경로의 일부를 끊고 재연결하여 거리를 줄이는 지역 탐색 알고리즘입니다.")
    c1, c2 = st.columns([3, 1])
    k_v = c1.radio("방식 선택", ["2-opt", "3-opt"], horizontal=True)
    timeout = c1.slider("실행 시간 제한 (초)", 1, 20, 5, key="kopt_time")
    timer_spot = c1.empty()
    graph_spot = st.empty()
    if c2.button("알고리즘 실행", key="kopt", type="primary", use_container_width=True):
        res, t = run_algorithm_in_background(algo.run_kopt, (k_v, st.session_state.cities, timeout), graph_spot, "green", timer_spot)
        if res:
            st.session_state.paths["k-opt"] = res
            st.session_state.scores["k-opt"] = algo.calculate_total_dist(res, st.session_state.cities)
            st.session_state.times["k-opt"] = t
            st.rerun()
    else: graph_spot.plotly_chart(draw_tsp_plot(st.session_state.cities, st.session_state.paths["k-opt"], "k-opt 결과", "green"), config=chart_config)

# 5. Metaheuristic
with tabs[4]:
    st.markdown("> **Metaheuristic**: 초기 해 생성 전략과 지역 탐색(Local Search) 전략을 조합하여 최적해를 탐색합니다.")
    c1, c2 = st.columns([3, 1])
    with c1:
        c1_1, c1_2 = st.columns(2)
        init_strategy = c1_1.selectbox("초기 해 생성 (Initialization)", ["Automatic", "Greedy", "Savings", "Sweep", "Christofides"], index=0)
        meta_strategy = c1_2.selectbox("지역 탐색 (Metaheuristic)", ["Automatic", "Greedy Descent", "Guided Local Search", "Simulated Annealing", "Tabu Search"], index=3)
        timeout = st.slider("실행 시간 제한 (초)", 1, 20, 5, key="meta_time")
        timer_spot = st.empty()
        temp_arg = None
        if meta_strategy == "Simulated Annealing":
            init_temp = st.number_input("초기 온도 (Initial Temp)", min_value=0, value=0, help="0으로 설정하면 OR-Tools가 자동으로 결정합니다.")
            temp_arg = init_temp if init_temp > 0 else None
    graph_spot = st.empty()
    if c2.button("알고리즘 실행", key="meta", type="primary", use_container_width=True):
        res, t = run_algorithm_in_background(algo.run_metaheuristic, (st.session_state.cities, timeout, init_strategy, meta_strategy, temp_arg), graph_spot, "purple", timer_spot)
        if res:
            st.session_state.paths["Metaheuristic"] = res
            st.session_state.scores["Metaheuristic"] = algo.calculate_total_dist(res, st.session_state.cities)
            st.session_state.times["Metaheuristic"] = t
            st.rerun()
    else: graph_spot.plotly_chart(draw_tsp_plot(st.session_state.cities, st.session_state.paths["Metaheuristic"], "결과", "purple"), config=chart_config)

# 6. Neural Network
with tabs[5]:
    st.markdown("""
    > **Neural Network (Pointer Network)**  
    > 미리 학습된 Pointer Network(PyTorch, `models/pointer_network.pt`)를 사용하여 TSP 경로를 추론합니다.  
    > 입력 좌표는 0~100 범위에서 0~1로 정규화되어 모델에 전달됩니다.
    """)

    c1, c2 = st.columns([3, 1])

    with c1:
        st.markdown("### 사용 모델: **Pre-trained Pointer Network**")
        timer_spot = st.empty()

    graph_spot = st.empty()

    # 리더보드에서 구분하기 위한 고정 키
    model_name_for_key = "Neural: Pointer Network"

    # 해당 모델의 기존 결과가 있으면 불러오기
    current_path = st.session_state.paths.get(model_name_for_key, [])

    if c2.button("알고리즘 실행", key="neural_btn", type="primary", use_container_width=True):
        # run_algorithm_in_background는 target_func(*args, callback=...) 형태로 호출함
        res, t = run_algorithm_in_background(
            algo.run_neural,
            (st.session_state.cities,),
            graph_spot, "magenta", timer_spot
        )
        if res:
            st.session_state.paths[model_name_for_key] = res
            st.session_state.scores[model_name_for_key] = algo.calculate_total_dist(res, st.session_state.cities)
            st.session_state.times[model_name_for_key] = t
            st.rerun()
    else:
        # 실행 버튼을 누르지 않았을 때는 기존 결과를 보여주거나 빈 그래프를 그림
        if current_path:
            graph_spot.plotly_chart(
                draw_tsp_plot(
                    st.session_state.cities,
                    current_path,
                    model_name_for_key,
                    "magenta"
                ),
                config=chart_config
            )
        else:
            graph_spot.plotly_chart(
                draw_tsp_plot(
                    st.session_state.cities,
                    [],
                    "Neural Pointer Network Result",
                    "magenta"
                ),
                config=chart_config
            )
