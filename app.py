import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# 1. 초기 설정
st.set_page_config(page_title="Interactive TSP Solver", layout="wide")
st.title("🧩 인터랙티브 TSP 알고리즘 시각화")

# 세션 상태 초기화
if 'cities' not in st.session_state:
    # 좌표 생성 후 소숫점 1자리 반올림 (피드백 1)
    coords = np.round(np.random.rand(10, 2) * 100, 1)
    st.session_state.cities = pd.DataFrame(coords, columns=['x', 'y'])
    st.session_state.user_path = []
    st.session_state.nn_path = []
    st.session_state.animating = False

def calculate_total_distance(path, cities_df):
    if len(path) < 2: return 0
    dist = 0
    # TSP 루프: 마지막에서 처음으로 돌아오는 거리 포함 (피드백 3)
    for i in range(len(path)):
        c1 = cities_df.iloc[path[i]]
        c2 = cities_df.iloc[path[(i + 1) % len(path)]] if i + 1 < len(path) else cities_df.iloc[path[0]]
        dist += np.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
    return dist

def reset_game():
    coords = np.round(np.random.rand(10, 2) * 100, 1)
    st.session_state.cities = pd.DataFrame(coords, columns=['x', 'y'])
    st.session_state.user_path = []
    st.session_state.nn_path = []
    st.session_state.animating = False
    st.rerun()

# 2. 사이드바 - 점수 및 제어
with st.sidebar:
    st.header("🎮 Control Panel")
    if st.button("새 게임 (도시 재생성)"):
        reset_game()
    
    st.divider()
    
    # 사용자 거리 계산
    user_dist = calculate_total_distance(st.session_state.user_path, st.session_state.cities)
    st.metric("나의 경로 거리", f"{user_dist:.1f}")
    st.write(f"방문 도시: {len(st.session_state.user_path)} / 10")
    
    # 알고리즘 거리 계산
    nn_dist = calculate_total_distance(st.session_state.nn_path, st.session_state.cities)
    st.metric("NN 알고리즘 거리", f"{nn_dist:.1f}", delta=f"{user_dist - nn_dist:.1f}" if nn_dist > 0 else None, delta_color="inverse")

    if st.button("🚀 Algorithm Start (NN)"):
        st.session_state.nn_path = []
        st.session_state.animating = True

# 3. 알고리즘 실행 로직 (피드백 4)
if st.session_state.animating:
    current_node = 0
    nn_path = [current_node]
    unvisited = list(range(1, 10))
    
    placeholder = st.empty() # 실시간 업데이트를 위한 공간
    
    while unvisited:
        last_node = nn_path[-1]
        last_coord = st.session_state.cities.iloc[last_node]
        
        # 가장 가까운 노드 찾기
        next_node = min(unvisited, key=lambda x: np.hypot(
            st.session_state.cities.iloc[x].x - last_coord.x,
            st.session_state.cities.iloc[x].y - last_coord.y
        ))
        
        nn_path.append(next_node)
        unvisited.remove(next_node)
        st.session_state.nn_path = nn_path
        time.sleep(0.5) # 애니메이션 속도 조절
        st.rerun()
    
    st.session_state.animating = False

# 4. 시각화 (Plotly)
fig = go.Figure()

# 도시 (점) 표시 - 소숫점 1자리로 툴팁 수정 (피드백 1)
fig.add_trace(go.Scatter(
    x=st.session_state.cities['x'],
    y=st.session_state.cities['y'],
    mode='markers+text',
    text=[f"City {i}" for i in range(10)],
    hovertemplate="<b>%{text}</b><br>X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>",
    textposition="top center",
    marker=dict(size=12, color='rgba(70, 130, 180, 0.6)', line=dict(width=2, color='DarkSlateGrey')),
    name="도시"
))

# 사용자 경로 (피드백 3: 루프 연결)
if len(st.session_state.user_path) > 0:
    indices = st.session_state.user_path
    if len(indices) == 10: # 모든 도시 방문 시 처음으로 연결
        indices = indices + [indices[0]]
    
    path_coords = st.session_state.cities.iloc[indices]
    fig.add_trace(go.Scatter(
        x=path_coords['x'], y=path_coords['y'],
        mode='lines+markers',
        line=dict(color='firebrick', width=3),
        name="나의 경로"
    ))

# NN 알고리즘 경로 (피드백 4)
if len(st.session_state.nn_path) > 0:
    nn_indices = st.session_state.nn_path
    if len(nn_indices) == 10:
        nn_indices = nn_indices + [nn_indices[0]]
        
    nn_coords = st.session_state.cities.iloc[nn_indices]
    fig.add_trace(go.Scatter(
        x=nn_coords['x'], y=nn_coords['y'],
        mode='lines',
        line=dict(color='rgba(0, 128, 0, 0.5)', width=5, dash='dot'),
        name="NN 알고리즘"
    ))

# 레이아웃 수정 (피드백 2: 가로선 제거)
fig.update_layout(
    template="plotly_white",
    xaxis=dict(showgrid=False, zeroline=False, range=[-10, 110]),
    yaxis=dict(showgrid=False, zeroline=False, range=[-10, 110]),
    height=700,
    margin=dict(l=20, r=20, t=20, b=20),
    clickmode='event+select'
)

# 5. 클릭 이벤트 처리
selected_points = st.plotly_chart(fig, on_select="rerun", key="tsp_chart", use_container_width=True)

if selected_points and "selection" in selected_points:
    indices = selected_points["selection"]["point_indices"]
    if indices:
        new_point = indices[0]
        if new_point not in st.session_state.user_path:
            st.session_state.user_path.append(new_point)
            st.rerun()

st.info("📍 도시를 순서대로 클릭하여 최단 경로를 만들어보세요! 모든 도시를 클릭하면 자동으로 출발점으로 연결됩니다.")
