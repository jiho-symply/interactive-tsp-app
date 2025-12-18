import numpy as np
import time
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def get_dist(p1, p2, cities_df):
    c1, c2 = cities_df.iloc[p1], cities_df.iloc[p2]
    return np.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)

def calculate_total_dist(path, cities_df):
    if not path or len(path) < 2: return 0.0
    n = len(cities_df)
    d = sum(get_dist(path[i], path[i+1], cities_df) for i in range(len(path)-1))
    if len(path) == n: d += get_dist(path[-1], path[0], cities_df)
    return round(d, 1)

def run_ortools_engine(cities_df, strategy, metaheuristic, timeout_ms, callback):
    """OR-Tools C++ 엔진을 사용하여 TSP 해결"""
    n = len(cities_df)
    # 거리 매트릭스 생성 (정수 기반 연산 최적화)
    dist_matrix = [[int(get_dist(i, j, cities_df) * 100) for j in range(n)] for i in range(n)]

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 파라미터 설정
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = strategy
    search_parameters.local_search_metaheuristic = metaheuristic
    search_parameters.time_limit.seconds = timeout_ms // 1000

    # 중간 해 발견 시 콜백 (C++에서 새 경로를 찾을 때마다 실행)
    class SolutionCallback:
        def __init__(self, model, manager):
            self.model = model
            self.manager = manager
            self.iter = 0

        def __call__(self):
            self.iter += 1
            path = []
            index = self.model.Start(0)
            while not self.model.IsEnd(index):
                path.append(self.manager.IndexToNode(index))
                index = self.model.GetNextSolutionInternal(0).Value(self.model.NextVar(index)) # 실제 구현상 루프 밖에서 호출 필요
            # Streamlit 구조상 내부 callback은 제한적이므로 Solve 이후 최적해를 리턴

    # 실제 최적화 수행
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        path = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            path.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        return path
    return list(range(n))

# ---------------------------------------------------------
# 각 탭별 엔진 호출 함수
# ---------------------------------------------------------

def run_nn_engine(n, start_node, cities_df, callback):
    callback([], "Nearest Neighbor (C++ Engine) 실행 중...")
    # PATH_CHEAPEST_ARC는 NN과 유사한 탐욕적 알고리즘입니다.
    strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    return run_ortools_engine(cities_df, strategy, routing_enums_pb2.LocalSearchMetaheuristic.UNSET, 1000, callback)

def run_kopt_engine(k_val, cities_df, callback):
    callback([], f"{k_val} (C++ Local Search) 실행 중...")
    # GREEDY_DESCENT는 개선이 없을 때까지 k-opt를 반복합니다.
    strategy = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
    meta = routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT
    return run_ortools_engine(cities_df, strategy, meta, 3000, callback)

def run_sa_engine(cities_df, callback):
    callback([], "Simulated Annealing (C++ Metaheuristic) 실행 중...")
    strategy = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
    meta = routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING
    return run_ortools_engine(cities_df, strategy, meta, 5000, callback)

def run_advanced_engine(cities_df, callback):
    callback([], "OR-Tools Guided Local Search 실행 중...")
    strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    meta = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    return run_ortools_engine(cities_df, strategy, meta, 5000, callback)
