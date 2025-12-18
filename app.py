import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import algorithms as algo  # 모듈 임포트

# --- 설정 및 세션 초기화 ---
st.set_page_config(page_title="TSP 시뮬레이터", layout="wide")
st.title("🏙️ TSP 시뮬레이터")

if 'n_cities' not in st.session_state: st.session_state.n_cities = 20
if 'cities' not in st.session_state:
    coords = np.round(np.random.rand(st.session_state.n_cities, 2) * 100, 1)
    st.session_state.cities = pd.DataFrame(coords, columns=['x', 'y'])
    st.session_state.paths = {k: [] for k in ["대학원생 최적화", "Nearest Neighbor", "k-opt", "Simulated Annealing"]}
    st.session_state.scores = {k: 0.0 for k in st.session_state.paths.keys()}

# --- 공용 시각화 함수 ---
def draw_tsp_plot(path, title, color="orange", key="plot"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=st.session_state.cities.x, y=st.session_state.cities.y,
        mode='markers+text', text=[f"C{i}" for i in range(st.session_state.n_cities)],
        textposition="top center", marker=dict(size=10, color='black'), name="도시"
    ))
    if path:
        d_path = path + [path[0]] if len(path) == st.session_state.n_cities else path
        coords = st.session_state.cities.iloc[d_path]
        fig.add_trace(go.Scatter(x=coords.x, y=coords.y, mode='lines+markers', line=dict(color=color, width=3)))
    
    fig.update_layout(
        template="plotly_white", xaxis=dict(showgrid=False, range=[-5, 105], constrain="domain"),
        yaxis=dict(showgrid=False, range=[-5, 105], scaleanchor="x", scaleratio=1),
        height=900, showlegend=False, title=f"{title} (거리: {algo.calculate_total_dist(path, st.session_state.cities)})"
    )
    return fig

# --- 메인 UI 레이아웃 ---
col_main, col_side = st.columns([3, 1])

with col_side:
    st.subheader("📊 결과 비교표")
    score_list = [{"모드": k, "거리": v} for k, v in st.session_state.scores.items() if v > 0]
    if score_list:
        df = pd.DataFrame(score_list).sort_values(by="거리").reset_index(drop=True)
        df.index += 1
        st.table(df.style.format({"거리": "{:.1f}"}))
    
    if st.button("🗺️ 새 도시 배치", use_container_width=True):
        st.session_state.clear()
        st.rerun()

with col_main:
    tabs = st.tabs(["✍️ 대학원생 최적화", "📍 Nearest Neighbor", "🔧 k-opt", "🔥 Simulated Annealing"])

    # 1. 대학원생 최적화
    with tabs[0]:
        st.info("💡 대학원생의 직관은 때론 휴리스틱보다 강력합니다. 점을 순서대로 클릭하여 경로를 설계하세요.")
        if st.button("🧹 경로 초기화"):
            st.session_state.paths["대학원생 최적화"] = []
            st.session_state.scores["대학원생 최적화"] = 0.0
            st.rerun()
        
        graph_spot = st.empty()
        selected = graph_spot.plotly_chart(draw_tsp_plot(st.session_state.paths["대학원생 최적화"], "사용자 설계", "orange", "grad"), on_select="rerun", use_container_width=True)
        
        if selected and "selection" in selected and selected["selection"]["point_indices"]:
            idx = selected["selection"]["point_indices"][0]
            path = st.session_state.paths["대학원생 최적화"]
            if idx in path: path.remove(idx)
            else: path.append(idx)
            st.session_state.scores["대학원생 최적화"] = algo.calculate_total_dist(path, st.session_state.cities)
            st.rerun()

    # 2. Nearest Neighbor
    with tabs[1]:
        st.markdown("> **Nearest Neighbor**: 가장 가까운 도시를 차례로 방문하는 탐욕 알고리즘입니다.")
        c1, c2 = st.columns(2)
        start_node = c1.selectbox("시작점", range(st.session_state.n_cities))
        graph_spot = st.empty()
        
        if c2.button("NN 실행", type="primary"):
            def nn_callback(p, t):
                graph_spot.plotly_chart(draw_tsp_plot(p, t, "royalblue", f"nn_{len(p)}"), use_container_width=True)
            
            res = algo.run_nn(st.session_state.n_cities, start_node, st.session_state.cities, nn_callback)
            st.session_state.paths["Nearest Neighbor"] = res
            st.session_state.scores["Nearest Neighbor"] = algo.calculate_total_dist(res, st.session_state.cities)
            st.rerun()
        else:
            graph_spot.plotly_chart(draw_tsp_plot(st.session_state.paths["Nearest Neighbor"], "NN 결과", "royalblue"), use_container_width=True)

    # 3. k-opt
    with tabs[2]:
        st.markdown("> **k-opt**: 간선을 교체하며 경로를 개선하는 지역 탐색 기법입니다.")
        c1, c2 = st.columns(2)
        k_val = c1.radio("선택", ["2-opt", "3-opt"], horizontal=True)
        graph_spot = st.empty()
        
        if c2.button("k-opt 실행", type="primary"):
            init = st.session_state.paths["Nearest Neighbor"] or list(range(st.session_state.n_cities))
            def kopt_callback(p, t):
                graph_spot.plotly_chart(draw_tsp_plot(p, t, "green", f"kopt_{time.time()}"), use_container_width=True)
            
            res = algo.run_kopt(k_val, st.session_state.n_cities, init, st.session_state.cities, kopt_callback)
            st.session_state.paths["k-opt"] = res
            st.session_state.scores["k-opt"] = algo.calculate_total_dist(res, st.session_state.cities)
            st.rerun()
        else:
            graph_spot.plotly_chart(draw_tsp_plot(st.session_state.paths["k-opt"], "k-opt 결과", "green"), use_container_width=True)

    # 4. Simulated Annealing
    with tabs[3]:
        st.markdown("> **Simulated Annealing**: 확률적으로 나쁜 해를 수용하며 전역 최적해를 찾는 담금질 기법입니다.")
        c1, c2 = st.columns(2)
        temp = c1.slider("온도", 10, 1000, 100)
        cool = c1.slider("냉각 속도", 0.9, 0.99, 0.98)
        graph_spot = st.empty()
        
        if c2.button("SA 실행", type="primary"):
            # 콜백 함수: 변경이 일어날 때만 호출됨
            def sa_callback(p, t, it):
                # DuplicateElementId 에러 방지를 위해 고유 key 부여
                graph_spot.plotly_chart(draw_tsp_plot(p, t, "purple", f"sa_{it}"), use_container_width=True)
            
            res = algo.run_sa(st.session_state.n_cities, temp, cool, st.session_state.cities, sa_callback)
            st.session_state.paths["Simulated Annealing"] = res
            st.session_state.scores["Simulated Annealing"] = algo.calculate_total_dist(res, st.session_state.cities)
            st.rerun()
        else:
            graph_spot.plotly_chart(draw_tsp_plot(st.session_state.paths["Simulated Annealing"], "SA 결과", "purple"), use_container_width=True)
