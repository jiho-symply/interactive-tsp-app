import numpy as np
import onnxruntime as ort
import os
import io

class NeuralTSPSolver:
    """
    ONNX 기반의 TSP Solver.
    파일 경로(str) 또는 업로드된 파일(bytes)을 받아 추론합니다.
    """
    def __init__(self, model_source):
        """
        Args:
            model_source: 파일 경로(str) 또는 바이너리 데이터(bytes)
        """
        try:
            # CPU 실행 Provider 설정
            providers = ['CPUExecutionProvider']
            
            # 1. 파일 경로인 경우
            if isinstance(model_source, str):
                if not os.path.exists(model_source):
                    # 더미 모드 (파일이 없을 때 테스트용)
                    self.session = None
                    return
                self.session = ort.InferenceSession(model_source, providers=providers)
            
            # 2. 업로드된 바이너리(bytes)인 경우
            else:
                self.session = ort.InferenceSession(model_source, providers=providers)

            # 입력/출력 노드 이름 추출
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
        except Exception as e:
            raise RuntimeError(f"ONNX 모델 로드 실패: {e}")

    def preprocess(self, cities_df):
        """
        입력 전처리: (N, 2) -> Normalized (1, N, 2)
        """
        coords = cities_df[['x', 'y']].values.astype(np.float32)
        
        # 0~100 좌표계를 0.0~1.0으로 정규화 (일반적인 Neural TSP 입력 형태)
        normalized_coords = coords / 100.0
        
        # 배치 차원 추가: [Batch_Size, Node, Features] -> [1, N, 2]
        input_tensor = np.expand_dims(normalized_coords, axis=0)
        return input_tensor

    def postprocess(self, output_tensor, n_cities):
        """
        출력 후처리: Tensor -> Path List
        """
        # 결과가 numpy array라면 리스트로 변환
        if isinstance(output_tensor, np.ndarray):
            # 보통 출력은 [Batch, Sequence] 형태의 인덱스
            path_indices = output_tensor[0].tolist()
        else:
            path_indices = list(output_tensor[0])
            
        # 정수형 변환 및 유효성 검사
        path = [int(idx) for idx in path_indices]
        
        # 만약 모델이 시작점을 마지막에 한 번 더 포함한다면 제거
        if len(path) > n_cities and path[0] == path[-1]:
            path = path[:-1]
            
        return path

    def solve(self, cities_df):
        # 세션이 없으면(더미 모드) 단순 순차 경로 반환
        if self.session is None:
            return list(range(len(cities_df)))

        # 1. 전처리
        input_data = self.preprocess(cities_df)
        
        # 2. 추론
        try:
            outputs = self.session.run([self.output_name], {self.input_name: input_data})
        except Exception as e:
            raise RuntimeError(f"추론 실행 중 오류 발생 (Input Shape 불일치 등): {e}")
        
        # 3. 후처리
        path = self.postprocess(outputs[0], len(cities_df))
        
        return path
