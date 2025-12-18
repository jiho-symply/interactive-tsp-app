import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# --- 1. 초기 설정 ---
st.set_page_config(page_title="TSP Graduate Student Edition", layout="wide")
st.title("🎓 TSP: 대학원생 최적화 시뮬레이터")
st.caption("알고리즘보다 뛰어난 대학원생의 '직관'과 '노동력'을 시험해보세요.")

if 'cities' not in st.session_state:
    # 도시 생성 (소수점 1자리)
    coords = np.round(np.random.rand(10, 2) * 100, 1)
    st.session_state.cities = pd.DataFrame(coords, columns=['x', 'y'])
    # 경로 저장소
    st.session_state.paths = {
        "대학원생 최적화": [],
        "Nearest Neighbor": [],
        "k-opt": []
    }
    st.session_state.scores = {"대학원생 최적화": 0.0, "Nearest Neighbor": 0.0, "k-opt": 0.0}

# 거리 계산 함수
def get_dist(p1, p2):
    c1, c2 = st.session_state.cities.iloc[p1], st.session_state.cities.iloc[p2]
    return np.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)

def total_dist(path):
    if len(path) < 2: return 0.0
    d = sum(get_dist(path[i], path[i+1]) for i in range(len(path)-1))
    if len(path) == 10: d += get_dist(path[-1], path[0])
    return round(d, 1)

# 공용 그래프 출력 함수
def draw_graph(path, title, color="royalblue"):
    fig = go.Figure()
    # 도시 점
    fig.add_trace(go.Scatter(
        x=st.session_state.cities.x, y=st.session_state.cities.y,
        mode='markers+text', text=[f"C{i}" for i in range(10)],
        textposition="top center", marker=dict(size=12, color='black'), name="도시"
    ))
    # 경로 선
    if path:
        display_path = path + [path[0]] if len(path) == 10 else path
        coords = st.session_state.cities.iloc[display_path]
        fig.add_trace(go.Scatter(
            x=coords.x, y=coords.y, mode='lines+markers',
            line=dict(color=color, width=3), name=title
        ))
    fig.update_layout(
        template="plotly_white", xaxis=dict(showgrid=False, range=[-10, 110]),
        yaxis=dict(showgrid=False, range=[-10, 110]), height=800, showlegend=False,
        title=f"실시간 렌더링: {title} (총 거리: {total_dist(path)})"
    )
    return fig

# --- 2. 메인 레이아웃 ---
col_main, col_side = st.columns([3, 1])

with col_side:
    st.subheader("🏆 연구 성과 (리더보드)")
    
    # [오류 해결] 데이터가 있을 때만 정렬 및 출력
    score_list = [{"모드": k, "거리": v} for k, v in st.session_state.scores.items() if v > 0]
    
    if score_list:
        score_df = pd.DataFrame(score_list).sort_values(by="거리")
        st.table(score_df)
    else:
        st.info("아직 연구 데이터가 없습니다.")
    
    if st.button("📑 연구 주제 변경 (새 게임)", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# --- 3. 탭 기반 알고리즘 컨트롤 및 그래프 ---
with col_main:
    tab1, tab2, tab3 = st.tabs(["✍️ 대학원생 최적화", "📍 Nearest Neighbor", "🔧 k-opt"])

    # --- Tab 1: 대학원생 최적화 (Manual Input) ---
    with tab1:
        st.info("💡 대학원생의 직관은 때론 휴리스틱보다 강력합니다. 점을 순서대로 클릭하여 경로를 설계하세요.")
        graph_spot1 = st.empty()
        fig1 = draw_graph(st.session_state.paths["대학원생 최적화"], "대학원생 최적화", "firebrick")
        selected = graph_spot1.plotly_chart(fig1, on_select="rerun", key="human_chart", use_container_width=True)
        
        if selected and "selection" in selected and selected["selection"]["point_indices"]:
            idx = selected["selection"]["point_indices"][0]
            if idx not in st.session_state.paths["대학원생 최적화"] and len(st.session_state.paths["대학원생 최적화"]) < 10:
                st.session_state.paths["대학원생 최적화"].append(idx)
                st.session_state.scores["대학원생 최적화"] = total_dist(st.session_state.paths["대학원생 최적화"])
                st.rerun()

    # --- Tab 2: Nearest Neighbor ---
    with tab2:
        st.markdown("### 탐욕적 탐색 (Greedy Search)")
        start_node = st.selectbox("실험 시작 도시 설정", range(10))
        graph_spot2 = st.empty()
        graph_spot2.plotly_chart(draw_graph(st.session_state.paths["Nearest Neighbor"], "Nearest Neighbor", "royalblue"), use_container_width=True)
        
        if st.button("시뮬레이션 시작", key="nn_btn"):
            path = [start_node]
            unvisited = [i for i in range(10) if i != start_node]
            while unvisited:
                last = path[-1]
                next_node = min(unvisited, key=lambda x: get_dist(last, x))
                path.append(next_node)
                unvisited.remove(next_node)
                
                st.session_state.paths["Nearest Neighbor"] = path
                st.session_state.scores["Nearest Neighbor"] = total_dist(path)
                graph_spot2.plotly_chart(draw_graph(path, "Nearest Neighbor", "royalblue"), use_container_width=True, key=f"nn_anim_{len(unvisited)}")
                time.sleep(0.3)
            st.rerun()

    # --- Tab 3: k-opt (2-opt & 3-opt) ---
    with tab3:
        st.markdown("### 반복적 개선 (Iterative Improvement)")
        k_val = st.radio("최적화 기법 선택", [2, 3], horizontal=True)
        max_iter = st.slider("최대 반복 횟수(Epochs)", 10, 200, 50)
        graph_spot3 = st.empty()
        
        init_path = st.session_state.paths["Nearest Neighbor"] if st.session_state.paths["Nearest Neighbor"] else list(range(10))
        graph_spot3.plotly_chart(draw_graph(st.session_state.paths["k-opt"] or init_path, f"{k_val}-opt 최적화", "green"), use_container_width=True)
        
        if st.button("Local Search 실행"):
            path = list(init_path)
            best_d = total_dist(path)
            log_spot = st.empty()
            
            for it in range(max_iter):
                improved = False
                if k_val == 2:
                    for i in range(1, 9):
                        for j in range(i+1, 10):
                            new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                            new_d = total_dist(new_path)
                            if new_d < best_d:
                                path, best_d = new_path, new_d
                                improved = True
                                break
                        if improved: break
                else: # 3-opt
                    for i in range(7):
                        for j in range(i+2, 9):
                            for k in range(j+2, 10):
                                # 3-way swap 중 한 케이스
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
                    graph_spot3.plotly_chart(draw_graph(path, f"{k_val}-opt 개선 중...", "green"), use_container_width=True, key=f"kopt_anim_{it}")
                    log_spot.success(f"📈 {it+1}회차: 거리 {best_d}로 단축됨!")
                    time.sleep(0.3)
                else:
                    log_spot.info(f"✨ 최적해 도달 (반복 {it+1}회)")
                    break
            st.rerun()
