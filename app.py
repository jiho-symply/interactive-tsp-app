import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# 초기 설정
st.set_page_config(page_title="Interactive TSP Solver", layout="wide")
st.title("🧩 인터랙티브 TSP 알고리즘 시각화")

# 1. 세션 상태 초기화
if 'cities' not in st.session_state:
    # 좌표 생성 및 소숫점 1자리 반올림
    coords = np.round(np.random.rand(10, 2) * 100, 1)
    st.session_state.cities = pd.DataFrame(coords, columns=['x', 'y'])
    st.session_state.user_path = []
    st.session_state.nn_path = []

def calculate_dist(p1_idx, p2_idx):
    c1 = st.session_state.cities.iloc[p1_idx]
    c2 = st.session_state.cities.iloc[p2_idx]
    return np.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)

def get_total_dist(path):
    if len(path) < 2: return 0
    d = sum(calculate_dist(path[i], path[i+1]) for i in range(len(path)-1))
    if len(path) == 10: # TSP 루프 완성 시
        d += calculate_dist(path[-1], path[0])
    return d

# 2. 그래프 그리기 함수 (재사용)
def draw_tsp_plot():
    fig = go.Figure()
    
    # 모든 도시 표시
    fig.add_trace(go.Scatter(
        x=st.session_state.cities['x'], y=st.session_state.cities['y'],
        mode='markers+text',
        text=[f"City {i}" for i in range(10)],
        hovertemplate="X: %{x}<br>Y: %{y}<extra></extra>",
        marker=dict(size=12, color='royalblue'),
        name="도시"
    ))

    # 사용자 경로
    if st.session_state.user_path:
        p = st.session_state.user_path
        if len(p) == 10: p = p + [p[0]]
        coords = st.session_state.cities.iloc[p]
        fig.add_trace(go.Scatter(x=coords.x, y=coords.y, mode='lines+markers',
                                 line=dict(color='firebrick', width=3), name="내 경로"))

    # NN 알고리즘 경로
    if st.session_state.nn_path:
        p = st.session_state.nn_path
        if len(p) == 10: p = p + [p[0]]
        coords = st.session_state.cities.iloc[p]
        fig.add_trace(go.Scatter(x=coords.x, y=coords.y, mode='lines',
                                 line=dict(color='rgba(0,128,0,0.4)', width=5, dash='dot'), name="NN"))

    fig.update_layout(
        template="plotly_white",
        xaxis=dict(showgrid=False, range=[-5, 105]),
        yaxis=dict(showgrid=False, range=[-5, 105]),
        height=600, clickmode='event+select', showlegend=False
    )
    return fig

# 3. 사이드바 구성
with st.sidebar:
    st.header("🎮 Control Panel")
    if st.button("새 게임 (도시 재생성)"):
        st.session_state.clear()
        st.rerun()
    
    user_d = get_total_dist(st.session_state.user_path)
    nn_d = get_total_dist(st.session_state.nn_path)
    
    st.metric("나의 경로 거리", f"{user_d:.1f}")
    st.metric("NN 알고리즘 거리", f"{nn_d:.1f}", 
              delta=f"{user_d - nn_d:.1f}" if nn_d > 0 else None, delta_color="inverse")

    start_algo = st.button("🚀 Algorithm Start (NN)")

# 4. 메인 화면 출력
st.info("📍 도시를 클릭하여 경로를 만드세요. 10개를 모두 선택하면 자동으로 시작점과 연결됩니다.")
plot_spot = st.empty() # 그래프가 들어갈 고정 자리

# 5. 알고리즘 실행 (버튼 클릭 시)
if start_algo:
    st.session_state.nn_path = [0]
    unvisited = list(range(1, 10))
    
    while unvisited:
        last = st.session_state.nn_path[-1]
        next_node = min(unvisited, key=lambda x: calculate_dist(last, x))
        st.session_state.nn_path.append(next_node)
        unvisited.remove(next_node)
        
        # 루프 내부에서 그래프만 즉시 업데이트
        plot_spot.plotly_chart(draw_tsp_plot(), use_container_width=True, key=f"nn_{len(unvisited)}")
        time.sleep(0.3)
    st.rerun() # 최종 상태 저장 및 UI 동기화

# 6. 사용자 인터랙션 처리 (평상시)
else:
    fig = draw_tsp_plot()
    selected = plot_spot.plotly_chart(fig, on_select="rerun", key="main_chart", use_container_width=True)

    if selected and "selection" in selected and selected["selection"]["point_indices"]:
        idx = selected["selection"]["point_indices"][0]
        if idx not in st.session_state.user_path and len(st.session_state.user_path) < 10:
            st.session_state.user_path.append(idx)
            st.rerun()
