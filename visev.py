# 시각화 & 평가지표 함수
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import networkx as nx
# 테스트 데이터셋 시각화
import matplotlib.patches as mpatches

def visualize_graph(data, model=None, threshold=0.5, title="Graph Visualization"):
    edge_list = data.edge_index.t().cpu().numpy()
    G = nx.Graph()
    G.add_edges_from(edge_list)
    pos = nx.spring_layout(G, seed=42)

    if model is None:
        labels = data.y.cpu().numpy()
        node_colors = ["red" if label == 1 else "blue" for label in labels]
        legend_elements = [
            mpatches.Patch(color='blue', label='Normal (0)'),
            mpatches.Patch(color='red', label='Anomaly (1)')
        ]
    else:
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > threshold).astype(int)
            true = data.y.cpu().numpy()
            node_colors = []
            for t, p in zip(true, preds):
                if t == 1 and p == 1:
                    node_colors.append("green")  # True Positive
                elif t == 0 and p == 0:
                    node_colors.append("blue")   # True Negative
                elif t == 0 and p == 1:
                    node_colors.append("orange") # False Positive
                else:
                    node_colors.append("red")    # False Negative

        # 범례 정의
        legend_elements = [
            mpatches.Patch(color='blue', label='True Negative (TN)'),
            mpatches.Patch(color='green', label='True Positive (TP)'),
            mpatches.Patch(color='orange', label='False Positive (FP)'),
            mpatches.Patch(color='red', label='False Negative (FN)')
        ]

    # 시각화
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=40, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    plt.title(title)
    plt.axis('off')
    plt.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()
    plt.savefig(f"./GraphM/graph_result/{title}.png")
    plt.show()
    plt.clf()
    plt.close()


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index).squeeze()
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        labels = data.y.cpu().numpy()
 

    print("예측 분포:", np.unique(preds, return_counts=True))
    print("실제 라벨 분포:", np.unique(labels, return_counts=True))

    # 여러 threshold에서 F1 점수 비교
    print("\nThreshold Tuning:")
    print("------------------")
    best_f1 = 0
    best_thresh = 0.5
    for threshold in np.arange(0.1, 0.9, 0.1):
        preds = (probs > threshold).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        print(f"Threshold {threshold:.1f} → F1 Score: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = threshold

    # 최적 threshold로 성능 평가
    final_preds = (probs > best_thresh).astype(int)
    auc_roc = roc_auc_score(labels, probs)
    auc_pr = average_precision_score(labels, probs)
    acc = accuracy_score(labels, final_preds)
    precision = precision_score(labels, final_preds, zero_division=0)
    recall = recall_score(labels, final_preds, zero_division=0)
    f1 = f1_score(labels, final_preds, zero_division=0)

    print("\nBest Threshold Evaluation")
    print("--------------------------")
    print(f"Best Threshold: {best_thresh:.2f}")
    print(f"AUC-ROC     : {auc_roc:.4f}")
    print(f"AUC-PR      : {auc_pr:.4f}")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"F1 Score    : {f1:.4f}")

    return {
        "best_threshold": best_thresh,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ✅ 고정 k-NN 그래프 생성
def build_euclidian_graph(dataframe, k=5):
    x = torch.tensor(dataframe.drop(columns=['class']).values, dtype=torch.float)
    y = torch.tensor(dataframe['class'].values, dtype=torch.float)

    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(x)
    _, indices = nbrs.kneighbors(x)

    edge_index = []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:
            edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, y=y)   #x, edge_index, y



# ✅ 고정 k-NN 그래프 생성
# 데이터 생성 예시 (노드 특성 및 엣지 정의)
def build_cosine_graph(dataframe, k=5):
    from sklearn.neighbors import NearestNeighbors

    x = torch.tensor(dataframe.drop(columns=['class']).values, dtype=torch.float)
    y = torch.tensor(dataframe['class'].values, dtype=torch.float)

    # k-NN을 통한 엣지 구성 (초기 그래프)
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(x)
    _, indices = nbrs.kneighbors(x)

    edge_index = []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:
            edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, y=y)


# 새로 추가함 3D로 시각화
# ✅ 먼저 이 셀을 실행해서 인터랙티브 모드 설정


# ✅ 3D 그래프 시각화 함수
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
import os

def visualize_graph_3d_rotate(data, title="3D Graph", save_dir="rotation_frames", angles=range(0, 360, 10)):
    os.makedirs(save_dir, exist_ok=True)
    
    edge_list = data.edge_index.t().cpu().numpy()
    G = nx.Graph()
    G.add_edges_from(edge_list)
    pos = nx.spring_layout(G, dim=3, seed=42)
    xyz = np.array([pos[i] for i in G.nodes()])

    labels = data.y.cpu().numpy()
    node_colors = ['red' if label == 1 else 'blue' for label in labels]

    for angle in angles:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for i, j in G.edges():
            x = [xyz[i, 0], xyz[j, 0]]
            y = [xyz[i, 1], xyz[j, 1]]
            z = [xyz[i, 2], xyz[j, 2]]
            ax.plot(x, y, z, color='gray', alpha=0.1)

        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=node_colors, s=20, alpha=0.8)
        ax.set_title(title + f" (angle={angle}°)")
        ax.set_axis_off()
        ax.view_init(elev=30, azim=angle)

        ax.legend(handles=[
            Patch(color='blue', label='Normal (0)'),
            Patch(color='red', label='Anomaly (1)')
        ], loc='upper right')

        plt.tight_layout()
        plt.savefig(f"{save_dir}/frame_{angle:03d}.png")
        plt.close()

    print(f"✅ Saved rotation frames to: {save_dir}/")


#azim=45 → 오른쪽 45도에서 본다
#elev=30 → 수평선보다 30도 위에서 내려다본다 (높은 시점)
def visualize_graph_3d_spherical_rotation(data, title="3D Graph", save_dir="rotation_sphere", steps=36):
    import os
    os.makedirs(save_dir, exist_ok=True)
    edge_list = data.edge_index.t().cpu().numpy()
    G = nx.Graph()
    G.add_edges_from(edge_list)
    pos = nx.spring_layout(G, dim=3, seed=42)
    xyz = np.array([pos[i] for i in G.nodes()])
    labels = data.y.cpu().numpy()
    node_colors = ['red' if label == 1 else 'blue' for label in labels]

    for i in range(steps):
        elev = 20 + 10 * np.sin(2 * np.pi * i / steps)       # 위아래로 흔들기
        azim = 360 * i / steps                                # 한 바퀴 돌기
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for u, v in G.edges():
            ax.plot(
                [xyz[u, 0], xyz[v, 0]],
                [xyz[u, 1], xyz[v, 1]],
                [xyz[u, 2], xyz[v, 2]],
                color='gray', alpha=0.1
            )

        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=node_colors, s=20, alpha=0.8)
        ax.set_title(f"{title} | elev={elev:.1f}, azim={azim:.1f}")
        ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)
        ax.legend(handles=[
            Patch(color='blue', label='Normal (0)'),
            Patch(color='red', label='Anomaly (1)')
        ], loc='upper right')

        plt.tight_layout()
        plt.savefig(f"{save_dir}/frame_{i:03d}.png")
        plt.close()

    print(f"✅ Spherical rotated images saved to {save_dir}/")
