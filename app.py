import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# --- 1. 초기 설정 및 유틸리티 ---
st.set_page_config(page_title="Advanced TSP Solver", layout="wide")
st.title("🏙️ TSP 최적화 벤치마크 시스템")

if 'cities' not in st.session_state:
    coords = np.round(np.random.rand(10, 2) * 100, 1)
    st.session_state.cities = pd.DataFrame(coords, columns=['x', 'y'])
    st.session_state.results = {
        "사용자": {"path": [], "dist": 0.0},
        "Nearest Neighbor": {"path": [], "dist": 0.0},
        "2-opt": {"path": [], "dist": 0.0, "log": []},
        "3-opt": {"path": [], "dist": 0.0, "log": []}
    }

def get_dist(p1_idx, p2_idx):
    c1 = st.session_state.cities.iloc[p1_idx]
    c2 = st.session_state.cities.iloc[p2_idx]
    return np.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)

def calculate_total_dist(path):
    if len(path) < 2: return 0.0
    d = sum(get_dist(path[i], path[i+1]) for i in range(len(path)-1))
    if len(path) == 10: d += get_dist(path[-1], path[0])
    return round(d, 1)

# --- 2. 알고리즘 구현 ---

def run_nearest_neighbor(start_node):
    path = [start_node]
    unvisited = [i for i in range(10) if i != start_node]
    while unvisited:
        last = path[-1]
        next_node = min(unvisited, key=lambda x: get_dist(last, x))
        path.append(next_node)
        unvisited.remove(next_node)
    return path

def run_2opt(initial_path, max_iter):
    path = list(initial_path) if initial_path else list(range(10))
    best_dist = calculate_total_dist(path)
    logs = [f"초기 거리: {best_dist}"]
    
    for _ in range(max_iter):
        improved = False
        for i in range(1, len(path) - 1):
            for j in range(i + 1, len(path)):
                new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                new_dist = calculate_total_dist(new_path)
                if new_dist < best_dist:
                    logs.append(f"교체: {path[i]}-{path[j]} 구간 반전 -> {new_dist}")
                    path = new_path
                    best_dist = new_dist
                    improved = True
        if not improved: break
    return path, best_dist, logs

def run_3opt(initial_path, max_iter):
    # 간소화된 3-opt 로직
    path = list(initial_path) if initial_path else list(range(10))
    best_dist = calculate_total_dist(path)
    logs = [f"초기 거리: {best_dist}"]
    
    for _ in range(max_iter):
        improved = False
        for i in range(len(path)):
            for j in range(i+2, len(path)):
                for k in range(j+2, len(path) + (1 if i > 0 else 0)):
                    # 3-opt는 여러 연결 조합이 있으나 여기서는 기본 교체 시도
                    new_path = path[:i] + path[j:k] + path[i:j] + path[k:]
                    new_dist = calculate_total_dist(new_path)
                    if new_dist < best_dist:
                        logs.append(f"3-way 교체 발견 -> {new_dist}")
                        path = new_path
                        best_dist = new_dist
                        improved = True
        if not improved: break
    return path, best_dist, logs

# --- 3. 메인 레이아웃 (상단: 그래프 & 점수판) ---
col1, col2 = st.columns([2, 1])

with col1:
    # 가시성 컨트롤
    visible_paths = st.multiselect("보여줄 경로 선택", 
                                   list(st.session_state.results.keys()), 
                                   default=["사용자"])
    
    fig = go.Figure()
    # 도시 점
    fig.add_trace(go.Scatter(
        x=st.session_state.cities.x, y=st.session_state.cities.y,
        mode='markers+text', text=[f"C{i}" for i in range(10)],
        textposition="top center", marker=dict(size=12, color='black'), name="도시"
    ))
    
    colors = {"사용자": "firebrick", "Nearest Neighbor": "royalblue", "2-opt": "green", "3-opt": "orange"}
    
    for name in visible_paths:
        res = st.session_state.results[name]
        if res["path"]:
            p = res["path"] + [res["path"][0]] if len(res["path"]) == 10 else res["path"]
            coords = st.session_state.cities.iloc[p]
            fig.add_trace(go.Scatter(x=coords.x, y=coords.y, mode='lines+markers',
                                     line=dict(color=colors[name], width=3 if name=="사용자" else 2), name=name))

    fig.update_layout(template="plotly_white", xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), height=500)
    
    # 클릭 이벤트 처리
    selected = st.plotly_chart(fig, on_select="rerun", key="main_chart", use_container_width=True)
    if selected and "selection" in selected and selected["selection"]["point_indices"]:
        idx = selected["selection"]["point_indices"][0]
        if idx not in st.session_state.results["사용자"]["path"] and len(st.session_state.results["사용자"]["path"]) < 10:
            st.session_state.results["사용자"]["path"].append(idx)
            st.session_state.results["사용자"]["dist"] = calculate_total_dist(st.session_state.results["사용자"]["path"])
            st.rerun()

with col2:
    st.subheader("🏆 Leaderboard")
    score_data = []
    for name, res in st.session_state.results.items():
        if res["dist"] > 0:
            score_data.append({"알고리즘": name, "거리": res["dist"]})
    
    if score_data:
        df = pd.DataFrame(score_data).sort_values(by="거리") # 오름차순 정렬
        st.table(df)
    else:
        st.write("알고리즘을 실행하여 결과를 확인하세요.")
    
    if st.button("게임 초기화", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# --- 4. 하단 탭 (알고리즘 컨트롤 패널) ---
st.divider()
tab_nn, tab_2opt, tab_3opt = st.tabs(["Nearest Neighbor", "2-opt", "3-opt"])

with tab_nn:
    start_city = st.selectbox("시작 도시 선택", range(10), key="nn_start")
    if st.button("NN 실행"):
        path = run_nearest_neighbor(start_city)
        st.session_state.results["Nearest Neighbor"]["path"] = path
        st.session_state.results["Nearest Neighbor"]["dist"] = calculate_total_dist(path)
        st.rerun()

with tab_2opt:
    iter_2 = st.slider("최대 반복 횟수", 10, 500, 100, key="2opt_iter")
    if st.button("2-opt 실행"):
        # NN 결과가 있다면 그것을 초기값으로, 없다면 0-9 순서로 시작
        init = st.session_state.results["Nearest Neighbor"]["path"]
        path, dist, logs = run_2opt(init, iter_2)
        st.session_state.results["2-opt"] = {"path": path, "dist": dist, "log": logs}
        st.rerun()
    if st.session_state.results["2-opt"]["log"]:
        st.text_area("2-opt 실행 로그", "\n".join(st.session_state.results["2-opt"]["log"]), height=150)

with tab_3opt:
    iter_3 = st.slider("최대 반복 횟수", 10, 500, 100, key="3opt_iter")
    if st.button("3-opt 실행"):
        init = st.session_state.results["2-opt"]["path"] or st.session_state.results["Nearest Neighbor"]["path"]
        path, dist, logs = run_3opt(init, iter_3)
        st.session_state.results["3-opt"] = {"path": path, "dist": dist, "log": logs}
        st.rerun()
    if st.session_state.results["3-opt"]["log"]:
        st.text_area("3-opt 실행 로그", "\n".join(st.session_state.results["3-opt"]["log"]), height=150)
