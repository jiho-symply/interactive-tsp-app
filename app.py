import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import algorithms as algo
import time

# --- 설정 및 세션 초기화 ---
st.set_page_config(page_title="TSP 시뮬레이터", layout="wide")
st.title("🏙️ TSP 시뮬레이터")

# [수정] 기본 도시 개수 25개 설정
if 'n_cities' not in st.session_state: st.session_state.n_cities = 25

if 'cities' not in st.session_state:
    coords = np.round(np.random.rand(st.session_state.n_cities, 2) * 100, 1)
    st.session_state.cities = pd.DataFrame(coords, columns=['x', 'y'])
    st.session_state.paths = {k: [] for k in ["대학원생 최적화", "Nearest Neighbor", "k-opt", "Simulated Annealing"]}
    st.session_state.scores = {k: 0.0 for k in st.session_state.paths.keys()}

# --- [수정] 새 도시 배치 다이얼로그 (새 창 형태) ---
@st.dialog("새 도시 배치")
def reset_cities_dialog():
    st.write("생성할 도시의 개수를 선택하세요. (최대 50개)")
    # [수정] 최대 도시 개수 50개 제한
    num = st.number_input("도시 개수", min_value=5, max_value=50, value=st.session_state.n_cities)
    
    c1, c2 = st.columns(2)
    if c1.button("취소", use_container_width=True):
        st.rerun()
    # [수정] '배치 생성' 버튼으로 명칭 변경 및 랜덤 좌표 생성 보장
    if c2.button("배치 생성", use_container_width=True, type="primary"):
        st.session_state.n_cities = num
        coords = np.round(np.random.rand(num, 2) * 100, 1)
        st.session_state.cities = pd.DataFrame(coords, columns=['x', 'y'])
        # 모든 상태 초기화
        st.session_state.paths = {k: [] for k in st.session_state.paths.keys()}
        st.session_state.scores = {k: 0.0 for k in st.session_state.paths.keys()}
        st.rerun()

# --- 공용 시각화 함수 ---
def draw_tsp_plot(path, title, color="orange"):
    fig = go.Figure()
    # [수정] 도시 번호를 1부터 붙임 (C1, C2, ...)
    fig.add_trace(go.Scatter(
        x=st.session_state.cities.x, y=st.session_state.cities.y,
        mode='markers+text', 
        text=[f"C{i+1}" for i in range(st.session_state.n_cities)],
        textposition="top center", 
        marker=dict(size=10, color='black'), 
        name="도시"
    ))
    
    if path:
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

# --- 메인 UI 레이아웃 ---
col_main, col_side = st.columns([3, 1])

with col_side:
    st.subheader("📊 결과 비교표")
    score_list = [{"모드": k, "거리": v} for k, v in st.session_state.scores.items() if v > 0]
    if score_list:
        df = pd.DataFrame(score_list).sort_values(by="거리").reset_index(drop=True)
        # [수정] 순위 표기 및 1부터 시작
        df.index += 1
        df.index.name = "순위"
        st.table(df.style.format({"거리": "{:.1f}"}))
    else:
        st.info("실험 데이터 없음")
    
    if st.button("🗺️ 새 도시 배치", use_container_width=True):
        reset_cities_dialog()

with col_main:
    tabs = st.tabs(["✍️ 대학원생 최적화", "📍 Nearest Neighbor", "🔧 k-opt", "🔥 Simulated Annealing"])

    # 1. 대학원생 최적화
    with tabs[0]:
        st.info("💡 대학원생의 직관은 때론 휴리스틱보다 강력합니다. 점을 순서대로 클릭하여 경로를 설계하세요.")
        c1, c2 = st.columns([3, 1])
        # [수정] 버튼 위치 우측 정렬 유지
        if c2.button("🧹 경로 초기화", use_container_width=True):
            st.session_state.paths["대학원생 최적화"] = []
            st.session_state.scores["대학원생 최적화"] = 0.0
            st.rerun()
        
        graph_spot = st.empty()
        selected = graph_spot.plotly_chart(
            draw_tsp_plot(st.session_state.paths["대학원생 최적화"], "사용자 설계", "orange"), 
            on_select="rerun", 
            use_container_width=True,
            config=chart_config
        )
        
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
        opt_col, btn_col = st.columns([3, 1])
        start_node = opt_col.selectbox("시작점 (번호 1~N 대응)", range(st.session_state.n_cities), format_func=lambda x: f"도시 {x+1}")
        graph_spot = st.empty()
        
        # [수정] 버튼 명칭 '알고리즘 실행' 통일 및 우측 배치
        if btn_col.button("알고리즘 실행", key="btn_nn", use_container_width=True, type="primary"):
            def nn_callback(p, t):
                graph_spot.plotly_chart(draw_tsp_plot(p, t, "royalblue"), use_container_width=True, config=chart_config)
            
            res = algo.run_nn(st.session_state.n_cities, start_node, st.session_state.cities, nn_callback)
            st.session_state.paths["Nearest Neighbor"] = res
            st.session_state.scores["Nearest Neighbor"] = algo.calculate_total_dist(res, st.session_state.cities)
            st.rerun()
        else:
            graph_spot.plotly_chart(draw_tsp_plot(st.session_state.paths["Nearest Neighbor"], "NN 결과", "royalblue"), use_container_width=True, config=chart_config)

    # 3. k-opt
    with tabs[2]:
        st.markdown("> **k-opt**: 간선을 교체하며 경로를 개선하는 지역 탐색 기법입니다.")
        opt_col, btn_col = st.columns([3, 1])
        k_val = opt_col.radio("k-opt 선택", ["2-opt", "3-opt"], horizontal=True)
        graph_spot = st.empty()
        
        # [수정] 버튼 명칭 '알고리즘 실행' 통일 및 우측 배치
        if btn_col.button("알고리즘 실행", key="btn_kopt", use_container_width=True, type="primary"):
            def kopt_callback(p, t):
                # DuplicateElementId 방지 위해 타임스탬프 활용은 내부적으로 자동 처리되나, 시각화 성능 위해 key 생략 가능
                graph_spot.plotly_chart(draw_tsp_plot(p, t, "green"), use_container_width=True, config=chart_config)
            
            # [수정] 실행할 때마다 초기 경로 재생성 로직이 포함된 run_kopt 호출
            res = algo.run_kopt(k_val, st.session_state.n_cities, st.session_state.cities, kopt_callback)
            st.session_state.paths["k-opt"] = res
            st.session_state.scores["k-opt"] = algo.calculate_total_dist(res, st.session_state.cities)
            st.rerun()
        else:
            graph_spot.plotly_chart(draw_tsp_plot(st.session_state.paths["k-opt"], "k-opt 결과", "green"), use_container_width=True, config=chart_config)

    # 4. Simulated Annealing
    with tabs[3]:
        st.markdown("> **Simulated Annealing**: 확률적으로 나쁜 해를 수용하며 전역 최적해를 찾는 담금질 기법입니다.")
        opt_col, btn_col = st.columns([3, 1])
        temp = opt_col.slider("초기 온도", 10, 1000, 100)
        cool = opt_col.slider("냉각 속도", 0.9, 0.99, 0.98)
        graph_spot = st.empty()
        
        # [수정] 버튼 명칭 '알고리즘 실행' 통일 및 우측 배치
        if btn_col.button("알고리즘 실행", key="btn_sa", use_container_width=True, type="primary"):
            def sa_callback(p, t, it):
                graph_spot.plotly_chart(draw_tsp_plot(p, t, "purple"), use_container_width=True, config=chart_config)
            
            res = algo.run_sa(st.session_state.n_cities, temp, cool, st.session_state.cities, sa_callback)
            st.session_state.paths["Simulated Annealing"] = res
            st.session_state.scores["Simulated Annealing"] = algo.calculate_total_dist(res, st.session_state.cities)
            st.rerun()
        else:
            graph_spot.plotly_chart(draw_tsp_plot(st.session_state.paths["Simulated Annealing"], "SA 결과", "purple"), use_container_width=True, config=chart_config)
