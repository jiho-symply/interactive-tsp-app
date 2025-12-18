import numpy as np
import time
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from ortools.sat.python import cp_model

# Streamlit 스레드 컨텍스트 주입을 위한 유틸리티
from streamlit.runtime.scriptrunner import add_script_run_ctx

def get_dist(p1, p2, cities_df):
    c1, c2 = cities_df.iloc[p1], cities_df.iloc[p2]
    return int(np.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2) * 100)

def calculate_total_dist(path, cities_df):
    if not path or len(path) < 2: return 0.0
    n = len(cities_df)
    d = 0.0
    for i in range(len(path)-1):
        c1, c2 = cities_df.iloc[path[i]], cities_df.iloc[path[i+1]]
        d += np.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
    c1, c2 = cities_df.iloc[path[-1]], cities_df.iloc[path[0]]
    d += np.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
    return round(d, 1)

# --- 1. Nearest Neighbor ---
def run_nn(n, start_node, cities_df, timeout, callback):
    path = [start_node]
    unvisited = set(range(n)) - {start_node}
    start_time = time.time()
    
    while unvisited:
        # 타임아웃 체크
        if time.time() - start_time > timeout:
            break

        last = path[-1]
        curr_coords = cities_df.iloc[last][['x', 'y']].values
        candidates = list(unvisited)
        cand_coords = cities_df.iloc[candidates][['x', 'y']].values
        dists = np.sum((cand_coords - curr_coords)**2, axis=1)
        next_node = candidates[np.argmin(dists)]
        
        path.append(next_node)
        unvisited.remove(next_node)
        
        callback(path, f"탐욕적 탐색 중... ({len(path)}/{n})")
        time.sleep(0.1) # [수정] 0.1초
    
    return path

# --- 2. OR-Tools Routing Engine (k-opt, SA) ---
def run_routing_engine(cities_df, strategy, metaheuristic, timeout, algorithm_name, ctx, callback):
    n = len(cities_df)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return get_dist(manager.IndexToNode(from_index), manager.IndexToNode(to_index), cities_df)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = strategy
    search_parameters.local_search_metaheuristic = metaheuristic
    # [수정] 타임아웃 적용 (초 단위)
    search_parameters.time_limit.seconds = int(timeout)

    # 콜백 클래스 내에서 컨텍스트 재주입
    def solution_callback():
        # [핵심] C++ 스레드에 Streamlit 컨텍스트 주입
        if ctx: add_script_run_ctx(ctx)
        
        path = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            path.append(manager.IndexToNode(index))
            index = routing.NextVar(index).Value()
        
        callback(path, f"{algorithm_name} 진행 중...")
        time.sleep(0.1) # [수정] 0.1초

    routing.AddAtSolutionCallback(solution_callback)
    
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        path = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            path.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        return path
    return list(range(n))

def run_kopt(k_val, cities_df, timeout, ctx, callback):
    strategy = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
    meta = routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT
    return run_routing_engine(cities_df, strategy, meta, timeout, k_val, ctx, callback)

def run_sa(cities_df, timeout, ctx, callback):
    strategy = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
    meta = routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING
    return run_routing_engine(cities_df, strategy, meta, timeout, "Simulated Annealing", ctx, callback)

# --- 3. CP-SAT Solver (MILP 최적해) ---
class ObjCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, arcs, n, ctx, callback):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.arcs = arcs
        self.n = n
        self.ctx = ctx # Streamlit 컨텍스트 저장
        self.callback = callback

    def OnSolutionCallback(self):
        # [핵심] 콜백 실행 시 컨텍스트 복구 (NoSessionContext 에러 해결)
        if self.ctx: add_script_run_ctx(self.ctx)
        
        next_node = {}
        for i, j, lit in self.arcs:
            if self.Value(lit):
                next_node[i] = j
        
        path = [0]
        curr = 0
        visited = {0}
        while len(path) < self.n:
            if curr in next_node:
                curr = next_node[curr]
                if curr in visited: break 
                visited.add(curr)
                path.append(curr)
            else:
                break
        
        self.callback(path, "MILP 최적해 탐색 중...")
        time.sleep(0.1) # [수정] 0.1초

def run_optimal_solver(cities_df, timeout, ctx, callback):
    n = len(cities_df)
    model = cp_model.CpModel()
    
    dist_matrix = [[get_dist(i, j, cities_df) for j in range(n)] for i in range(n)]
    arcs = []
    for i in range(n):
        for j in range(n):
            if i != j:
                arcs.append((i, j, model.NewBoolVar(f'e_{i}_{j}')))

    model.AddCircuit([(i, j, lit) for (i, j, lit) in arcs])
    model.Minimize(sum(dist_matrix[i][j] * lit for (i, j, lit) in arcs))

    solver = cp_model.CpSolver()
    # [수정] 타임아웃 적용
    solver.parameters.max_time_in_seconds = float(timeout)
    
    # 컨텍스트 전달
    solution_callback = ObjCallback(arcs, n, ctx, callback)
    status = solver.Solve(model, solution_callback)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        next_node = {}
        for i, j, lit in arcs:
            if solver.Value(lit):
                next_node[i] = j
        path = [0]
        curr = 0
        while len(path) < n:
            curr = next_node[curr]
            path.append(curr)
        return path
    return []
