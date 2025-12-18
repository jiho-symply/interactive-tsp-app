import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. 초기 설정 및 데이터 생성
st.set_page_config(page_title="Interactive TSP Solver", layout="wide")
st.title("🧩 직접 풀어보는 TSP (외판원 문제)")

if 'cities' not in st.session_state:
    # 10개의 무작위 도시 생성
    coords = np.random.rand(10, 2) * 100
    st.session_state.cities = pd.DataFrame(coords, columns=['x', 'y'])
    st.session_state.path = []  # 사용자가 선택한 도시 인덱스 순서

def reset_game():
    st.session_state.path = []
    st.rerun()

# 2. 사이드바 컨트롤
with st.sidebar:
    st.header("설정 및 상태")
    if st.button("게임 초기화"):
        reset_game()
    
    st.write(f"방문한 도시 수: {len(st.session_state.path)} / 10")
    
    # 거리 계산 로직
    if len(st.session_state.path) > 1:
        dist = 0
        for i in range(len(st.session_state.path) - 1):
            c1 = st.session_state.cities.iloc[st.session_state.path[i]]
            c2 = st.session_state.cities.iloc[st.session_state.path[i+1]]
            dist += np.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
        st.metric("현재 총 거리", f"{dist:.2f}")

# 3. 메인 시각화 (Plotly)
fig = go.Figure()

# 모든 도시 표시
fig.add_trace(go.Scatter(
    x=st.session_state.cities['x'],
    y=st.session_state.cities['y'],
    mode='markers+text',
    text=[f"City {i}" for i in range(10)],
    textposition="top center",
    marker=dict(size=12, color='royalblue'),
    name="도시"
))

# 선택된 경로 표시
if len(st.session_state.path) > 0:
    path_coords = st.session_state.cities.iloc[st.session_state.path]
    fig.add_trace(go.Scatter(
        x=path_coords['x'],
        y=path_coords['y'],
        mode='lines+markers',
        line=dict(color='firebrick', width=3),
        marker=dict(size=15, color='orange'),
        name="내 경로"
    ))

fig.update_layout(
    clickmode='event+select',
    width=800, height=600,
    xaxis=dict(range=[-5, 105]), yaxis=dict(range=[-5, 105]),
    showlegend=False
)

# 4. 클릭 이벤트 처리 (Streamlit 1.35+ 버전의 신기능 활용)
selected_points = st.plotly_chart(fig, on_select="rerun", key="tsp_chart")

if selected_points and "selection" in selected_points:
    indices = selected_points["selection"]["point_indices"]
    if indices:
        new_point = indices[0]
        # 이미 선택된 점이 아니면 경로에 추가
        if new_point not in st.session_state.path:
            st.session_state.path.append(new_point)
            st.rerun()

st.info("💡 위 차트에서 도시(파란 점)를 순서대로 클릭하여 경로를 연결해보세요!")
