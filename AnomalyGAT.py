import torch
import torch.nn as nn
import torch.nn.functional as F
from visev import *
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# GNN 모델 정의 (GAT(Graph Attention Convolution) 기반)
class AnomalyGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2):
        super(AnomalyGAT, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.3)
        self.gat2 = GATConv(hidden_channels*heads, hidden_channels, heads=heads, dropout=0.3)
        #self.gat3 = GATConv(hidden_channels*heads, hidden_channels, heads=heads, dropout=0.3)
        self.gat4 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        #x = self.gat3(x, edge_index)
        #x = F.elu(x)
        x = self.gat4(x, edge_index)
        return x

# 훈련 루프 (Early Stopping 포함)
def train_c(model, data_train, data_val, epochs=150, lr=0.01, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data_train.x, data_train.edge_index).squeeze()
        loss = criterion(out, data_train.y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(data_val.x, data_val.edge_index).squeeze()
            val_loss = criterion(val_out, data_val.y)
            val_preds = torch.sigmoid(val_out).cpu().numpy()
            val_labels = data_val.y.cpu().numpy()
            val_auc = roc_auc_score(val_labels, val_preds)

        #print(f"Epoch {epoch:03d}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val AUC: {val_auc:.4f}")

        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
            
    model_path= os.path.join(save_dir, "GAT_best_model_cosine")

    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), model_path)
    return model

# 테스트 데이터셋 시각화
def visualize_graph(data, title="Graph Visualization"):
    edge_list = data.edge_index.t().cpu().numpy()
    G = nx.Graph()
    G.add_edges_from(edge_list)
    pos = nx.spring_layout(G, seed=42)
    labels = data.y.cpu().numpy()
    node_colors = ["red" if label == 1 else "blue" for label in labels]

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50, alpha=0.7)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.title(title)
    plt.axis('off')
    plt.show()
