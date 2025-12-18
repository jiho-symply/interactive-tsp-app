import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# --- 1. 초기 설정 ---
st.set_page_config(page_title="TSP 시뮬레이터", layout="wide")
st.title("🏙️ TSP 시뮬레이터")
st.caption("알고리즘과 대학원생의 직관을 비교하는 최적화 실험실입니다.")

# 세션 상태 초기화
if 'n_cities' not in st.session_state:
    st.session_state.n_cities = 20

if 'cities' not in st.session_state:
    coords = np.round(np.random.rand(st.session_state.n_cities, 2) * 100, 1)
    st.session_state.cities = pd.DataFrame(coords, columns=['x', 'y'])
    st.session_state.paths = {"대학원생 최적화": [], "Nearest Neighbor": [], "k-opt": []}
    st.session_state.scores = {"대학원생 최적화": 0.0, "Nearest Neighbor": 0.0, "k-opt": 0.0}

# --- 2. 다이얼로그 및 유틸리티 ---

@st.dialog("새 도시 배치")
def reset_cities_dialog():
    st.write("생성할 도시의 개수를 선택하세요.")
    num = st.number_input("도시 개수", min_value=5, max_value=100, value=st.session_state.n_cities)
    c1, c2 = st.columns(2)
    if c1.button("취소", use_container_width=True):
        st.rerun()
    if c2.button("생성", use_container_width=True, type="primary"):
        st.session_state.n_cities = num
        coords = np.round(np.random.rand(num, 2) * 100, 1)
        st.session_state.cities = pd.DataFrame(coords, columns=['x', 'y'])
        st.session_state.paths = {"대학원생 최적화": [], "Nearest Neighbor": [], "k-opt": []}
        st.session_state.scores = {"대학원생 최적화": 0.0, "Nearest Neighbor": 0.0, "k-opt": 0.0}
        st.rerun()

def get_dist(p1, p2):
    c1, c2 = st.session_state.cities.iloc[p1], st.session_state.cities.iloc[p2]
    return np.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)

def total_dist(path):
    n = st.session_state.n_cities
    if len(path) < 2: return 0.0
    d = sum(get_dist(path[i], path[i+1]) for i in range(len(path)-1))
    if len(path) == n: d += get_dist(path[-1], path[0])
    return round(d, 1)

def draw_graph(path, title, color="royalblue"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=st.session_state.cities.x, y=st.session_state.cities.y,
        mode='markers+text', text=[f"C{i}" for i in range(st.session_state.n_cities)],
        textposition="top center", marker=dict(size=10, color='black'), name="도시"
    ))
    if path:
        display_path = path + [path[0]] if len(path) == st.session_state.n_cities else path
        coords = st.session_state.cities.iloc[display_path]
        fig.add_trace(go.Scatter(x=coords.x, y=coords.y, mode='lines+markers', line=dict(color=color, width=2.5)))
    
    # 가로 세로 비율 1:1 강제 제한
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(showgrid=False, range=[-5, 105], constrain="domain"),
        yaxis=dict(showgrid=False, range=[-5, 105], scaleanchor="x", scaleratio=1),
        height=900, showlegend=False, title=f"{title} (거리: {total_dist(path)})"
    )
    return fig

# --- 3. 메인 레이아웃 ---
col_main, col_side = st.columns([3, 1])

with col_side:
    st.subheader("🏆 연구 성과")
    score_list = [{"모드": k, "거리": v} for k, v in st.session_state.scores.items() if v > 0]
    if score_list:
        st.table(pd.DataFrame(score_list).sort_values(by="거리"))
    else:
        st.info("실험 데이터 없음")
    
    if st.button("🗺️ 새 도시 배치", use_container_width=True):
        reset_cities_dialog()

with col_main:
    tab1, tab2, tab3 = st.tabs(["✍️ 대학원생 최적화", "📍 Nearest Neighbor", "🔧 k-opt"])

    # --- Tab 1: 대학원생 최적화 ---
    with tab1:
        st.info("도시를 순서대로 클릭하세요.")
        graph_spot1 = st.empty()
        fig1 = draw_graph(st.session_state.paths["대학원생 최적화"], "대학원생 최적화", "firebrick")
        selected = graph_spot1.plotly_chart(fig1, on_select="rerun", key="human_chart", use_container_width=True)
        
        if selected and "selection" in selected and selected["selection"]["point_indices"]:
            idx = selected["selection"]["point_indices"][0]
            if idx not in st.session_state.paths["대학원생 최적화"] and len(st.session_state.paths["대학원생 최적화"]) < st.session_state.n_cities:
                st.session_state.paths["대학원생 최적화"].append(idx)
                st.session_state.scores["대학원생 최적화"] = total_dist(st.session_state.paths["대학원생 최적화"])
                st.rerun()

    # --- Tab 2: Nearest Neighbor ---
    with tab2:
        opt_col, btn_col = st.columns(2)
        start_node = opt_col.selectbox("시작 도시", range(st.session_state.n_cities))
        graph_spot2 = st.empty()
        
        if btn_col.button("시뮬레이션 시작", use_container_width=True, type="primary"):
            path = [start_node]
            unvisited = [i for i in range(st.session_state.n_cities) if i != start_node]
            while unvisited:
                last = path[-1]
                next_node = min(unvisited, key=lambda x: get_dist(last, x))
                path.append(next_node)
                unvisited.remove(next_node)
                st.session_state.paths["Nearest Neighbor"] = path
                st.session_state.scores["Nearest Neighbor"] = total_dist(path)
                graph_spot2.plotly_chart(draw_graph(path, "Nearest Neighbor", "royalblue"), use_container_width=True, key=f"nn_a_{len(unvisited)}")
                time.sleep(0.1)
            st.rerun()
        else:
            graph_spot2.plotly_chart(draw_graph(st.session_state.paths["Nearest Neighbor"], "Nearest Neighbor", "royalblue"), use_container_width=True)

    # --- Tab 3: k-opt ---
    with tab3:
        opt_col, btn_col = st.columns(2)
        k_val = opt_col.radio("k-opt 선택", [2, 3], horizontal=True)
        graph_spot3 = st.empty()
        
        init_path = st.session_state.paths["Nearest Neighbor"] if st.session_state.paths["Nearest Neighbor"] else list(range(st.session_state.n_cities))
        
        if btn_col.button("Local Search 실행", use_container_width=True, type="primary"):
            path = list(init_path)
            best_d = total_dist(path)
            log_spot = st.empty()
            n = st.session_state.n_cities
            
            while True: # 최적화될 때까지 무한 반복
                improved = False
                if k_val == 2:
                    for i in range(1, n - 1):
                        for j in range(i + 1, n):
                            new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                            new_d = total_dist(new_path)
                            if new_d < best_d:
                                path, best_d = new_path, new_d
                                improved = True
                                break
                        if improved: break
                else: # 3-opt (간소화된 3-way swap)
                    for i in range(n - 4):
                        for j in range(i + 2, n - 2):
                            for k in range(j + 2, n):
                                new_path = path[:i] + path[j:k] + path[i:j] + path[k:]
                                new_d = total_dist(new_path)
                                if new_d < best_d:
                                    path, best_d = new_path, new_d
                                    improved = True
                                    break
                            if improved: break
                        if improved: break
                
                if improved:
                    st.session_state.paths["k-opt"] = path
                    st.session_state.scores["k-opt"] = best_d
                    graph_spot3.plotly_chart(draw_graph(path, f"{k_val}-opt 개선 중...", "green"), use_container_width=True)
                    log_spot.success(f"📈 개선됨: {best_d}")
                    time.sleep(0.05)
                else:
                    log_spot.info(f"✨ 지역 최적해(Local Optimum) 도달")
                    break
            st.rerun()
        else:
            graph_spot3.plotly_chart(draw_graph(st.session_state.paths["k-opt"] or init_path, f"{k_val}-opt", "green"), use_container_width=True)
