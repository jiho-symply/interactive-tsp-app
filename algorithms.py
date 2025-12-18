import numpy as np
import pandas as pd
import time

def get_dist(p1, p2, cities_df):
    c1, c2 = cities_df.iloc[p1], cities_df.iloc[p2]
    return np.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)

def calculate_total_dist(path, cities_df):
    if len(path) < 2: return 0.0
    n = len(cities_df)
    d = sum(get_dist(path[i], path[i+1], cities_df) for i in range(len(path)-1))
    if len(path) == n: d += get_dist(path[-1], path[0], cities_df)
    return round(d, 1)

def run_nn(n_cities, start_node, cities_df, callback):
    path = [start_node]
    unvisited = [i for i in range(n_cities) if i != start_node]
    while unvisited:
        last = path[-1]
        next_node = min(unvisited, key=lambda x: get_dist(last, x, cities_df))
        path.append(next_node)
        unvisited.remove(next_node)
        callback(path, "Nearest Neighbor")
        time.sleep(0.05)
    return path

def run_kopt(k_val, n, cities_df, callback):
    # [수정] 실행할 때마다 초기 경로 무작위 재생성
    path = list(range(n))
    np.random.shuffle(path)
    best_d = calculate_total_dist(path, cities_df)
    
    while True:
        improved = False
        if k_val == "2-opt":
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                    new_d = calculate_total_dist(new_path, cities_df)
                    if new_d < best_d:
                        path, best_d = new_path, new_d
                        improved = True; break
                if improved: break
        else: # 3-opt
            for i in range(n - 4):
                for j in range(i + 2, n - 2):
                    for k in range(j + 2, n):
                        new_path = path[:i] + path[j:k] + path[i:j] + path[k:]
                        new_d = calculate_total_dist(new_path, cities_df)
                        if new_d < best_d:
                            path, best_d = new_path, new_d
                            improved = True; break
                    if improved: break
                if improved: break
        
        if improved:
            callback(path, f"{k_val} 개선 중...")
            time.sleep(0.02)
        else: break
    return path

def run_sa(n, temp, cooling, cities_df, callback):
    path = list(range(n))
    np.random.shuffle(path)
    curr_d = calculate_total_dist(path, cities_df)
    best_path, best_d = list(path), curr_d
    T = float(temp)
    iteration = 0
    
    while T > 0.1:
        iteration += 1
        i, j = np.random.choice(range(n), 2, replace=False)
        new_path = list(path)
        new_path[i], new_path[j] = new_path[j], new_path[i]
        new_d = calculate_total_dist(new_path, cities_df)
        
        delta = new_d - curr_d
        accepted = False
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            path, curr_d = new_path, new_d
            accepted = True
            if curr_d < best_d:
                best_path, best_d = list(path), curr_d

        if accepted:
            callback(best_path, f"SA 최적화 (T={T:.1f})", iteration)
        T *= cooling
    return best_path
