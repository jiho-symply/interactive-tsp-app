import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# --- 1. 초기 설정 ---
st.set_page_config(page_title="TSP 시뮬레이터", layout="wide")
st.title("🏙️ TSP 시뮬레이터")

if 'n_cities' not in st.session_state:
    st.session_state.n_cities = 20

if 'cities' not in st.session_state:
    coords = np.round(np.random.rand(st.session_state.n_cities, 2) * 100, 1)
    st.session_state.cities = pd.DataFrame(coords, columns=['x', 'y'])
    st.session_state.paths = {
        "대학원생 최적화": [], 
        "Nearest Neighbor": [], 
        "k-opt": [],
        "Simulated Annealing": []
    }
    st.session_state.scores = {
        "대학원생 최적화": 0.0, 
        "Nearest Neighbor": 0.0, 
        "k-opt": 0.0,
        "Simulated Annealing": 0.0
    }

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
        st.session_state.paths = {k: [] for k in st.session_state.paths.keys()}
        st.session_state.scores = {k: 0.0 for k in st.session_state.scores.keys()}
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

def draw_graph(path, title, color="orange"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=st.session_state.cities.x, y=st.session_state.cities.y,
        mode='markers+text', text=[f"C{i}" for i in range(st.session_state.n_cities)],
        textposition="top center", marker=dict(size=10, color='black'), name="도시"
    ))
    if path:
        display_path = path + [path[0]] if len(path) == st.session_state.n_cities else path
        coords = st.session_state.cities.iloc[display_path]
        fig.add_trace(go.Scatter(x=coords.x, y=coords.y, mode='lines+markers', line=dict(color=color, width=3)))
    
    # 높이 900으로 확대 및 비율 1:1 유지
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
    st.subheader("📊 결과 비교표")
    score_list = [{"모드": k, "거리": v} for k, v in st.session_state.scores.items() if v > 0]
    if score_list:
        df = pd.DataFrame(score_list).sort_values(by="거리")
        df.insert(0, "순위", range(1, len(df) + 1)) # 순위 컬럼 추가
        st.table(df.style.format({"거리": "{:.1f}"})) # 소수점 1자리 표기
    else:
        st.info("실험 데이터 없음")
    
    if st.button("🗺️ 새 도시 배치", use_container_width=True):
        reset_cities_dialog()

with col_main:
    tabs = st.tabs(["✍️ 대학원생 최적화", "📍 Nearest Neighbor", "🔧 k-opt", "🔥 Simulated Annealing"])

    # --- Tab 1: 대학원생 최적화 ---
    with tabs[0]:
        st.info("💡 대학원생의 직관은 때론 휴리스틱보다 강력합니다. 점을 순서대로 클릭하여 경로를 설계하세요.")
        c1, c2 = st.columns([3, 1])
        if c2.button("🧹 경로 초기화", use_container_width=True):
            st.session_state.paths["대학원생 최적화"] = []
            st.session_state.scores["대학원생 최적화"] = 0.0
            st.rerun()
            
        graph_spot1 = st.empty()
        fig1 = draw_graph(st.session_state.paths["대학원생 최적화"], "대학원생 최적화", "orange")
        selected = graph_spot1.plotly_chart(fig1, on_select="rerun", key="human_chart", use_container_width=True)
        
        if selected and "selection" in selected and selected["selection"]["point_indices"]:
            idx = selected["selection"]["point_indices"][0]
            path = st.session_state.paths["대학원생 최적화"]
            if idx in path:
                path.remove(idx) # 이미 있으면 제거 (피드백 5)
            elif len(path) < st.session_state.n_cities:
                path.append(idx)
            
            st.session_state.paths["대학원생 최적화"] = path
            st.session_state.scores["대학원생 최적화"] = total_dist(path)
            st.rerun()

    # --- Tab 2: Nearest Neighbor ---
    with tabs[1]:
        st.markdown("> **Nearest Neighbor**: 현재 위치에서 가장 가까운 미방문 도시를 선택하며 경로를 확장하는 탐욕적(Greedy) 알고리즘입니다.")
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
                graph_spot2.plotly_chart(draw_graph(path, "Nearest Neighbor", "royalblue"), use_container_width=True)
                time.sleep(0.05)
            st.rerun()
        else:
            graph_spot2.plotly_chart(draw_graph(st.session_state.paths["Nearest Neighbor"], "Nearest Neighbor", "royalblue"), use_container_width=True)

    # --- Tab 3: k-opt ---
    with tabs[2]:
        st.markdown("> **k-opt**: 기존 경로에서 k개의 간선을 제거하고 새로운 방식으로 연결하여 경로를 지속적으로 개선하는 지역 탐색(Local Search) 알고리즘입니다.")
        opt_col, btn_col = st.columns(2)
        k_val = opt_col.radio("k-opt 선택", ["2-opt", "3-opt"], horizontal=True)
        
        c1, c2 = btn_col.columns(2)
        if c1.button("최적화 시작", use_container_width=True, type="primary"):
            # NN 결과가 있으면 사용, 없으면 순차
            path = st.session_state.paths["Nearest Neighbor"] if st.session_state.paths["Nearest Neighbor"] else list(range(st.session_state.n_cities))
            best_d = total_dist(path)
            n = st.session_state.n_cities
            graph_spot3 = st.empty()
            
            while True:
                improved = False
                if k_val == "2-opt":
                    for i in range(1, n - 1):
                        for j in range(i + 1, n):
                            new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                            new_d = total_dist(new_path)
                            if new_d < best_d:
                                path, best_d = new_path, new_d
                                improved = True; break
                        if improved: break
                else: # 3-opt
                    for i in range(n - 4):
                        for j in range(i + 2, n - 2):
                            for k in range(j + 2, n):
                                new_path = path[:i] + path[j:k] + path[i:j] + path[k:]
                                new_d = total_dist(new_path)
                                if new_d < best_d:
                                    path, best_d = new_path, new_d
                                    improved = True; break
                            if improved: break
                        if improved: break
                
                if improved:
                    st.session_state.paths["k-opt"] = path
                    st.session_state.scores["k-opt"] = best_d
                    graph_spot3.plotly_chart(draw_graph(path, f"{k_val} 개선 중...", "green"), use_container_width=True)
                    time.sleep(0.02)
                else: break
            st.rerun()
        
        if c2.button("🔄 초기 경로 생성", use_container_width=True):
            st.session_state.paths["k-opt"] = []
            st.session_state.scores["k-opt"] = 0.0
            st.rerun()
            
        st.plotly_chart(draw_graph(st.session_state.paths["k-opt"], "k-opt 결과", "green"), use_container_width=True)

    # --- Tab 4: Simulated Annealing ---
    with tabs[3]:
        st.markdown("> **Simulated Annealing**: 금속의 담금질 과정을 모방한 알고리즘입니다. 초기에는 나쁜 해라도 일정 확률로 수용하여 지역 최적해(Local Optimum)를 탈출하고 전역 최적해를 찾습니다.")
        opt_col, btn_col = st.columns(2)
        temp = opt_col.slider("초기 온도", 10, 1000, 100)
        cooling = opt_col.slider("냉각 속도", 0.90, 0.99, 0.98)
        
        if btn_col.button("SA 시뮬레이션 시작", use_container_width=True, type="primary"):
            path = list(range(st.session_state.n_cities))
            np.random.shuffle(path)
            curr_d = total_dist(path)
            best_path, best_d = list(path), curr_d
            T = float(temp)
            graph_spot4 = st.empty()
            
            while T > 0.1:
                # 무작위 두 도시 교체
                i, j = np.random.choice(range(len(path)), 2, replace=False)
                new_path = list(path)
                new_path[i], new_path[j] = new_path[j], new_path[i]
                new_d = total_dist(new_path)
                
                delta = new_d - curr_d
                if delta < 0 or np.random.rand() < np.exp(-delta / T):
                    path, curr_d = new_path, new_d
                    if curr_d < best_d:
                        best_path, best_d = list(path), curr_d
                
                T *= cooling
                st.session_state.paths["Simulated Annealing"] = best_path
                st.session_state.scores["Simulated Annealing"] = best_d
                graph_spot4.plotly_chart(draw_graph(best_path, f"SA 최적화 (T={T:.1f})", "purple"), use_container_width=True)
            st.rerun()
        else:
            st.plotly_chart(draw_graph(st.session_state.paths["Simulated Annealing"], "SA 결과", "purple"), use_container_width=True)
