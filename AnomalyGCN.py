import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from visev import *
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import networkx as nx
import os  # ✅ 디렉토리 생성용

# ✅ 결과 저장 경로
save_dir = "./GraphM/graph_result"
os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없으면 생성

# ✅ GCN 모델 정의
class AnomalyGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(AnomalyGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# ✅ 훈련 함수 with early stopping
def train_gcn(model, data_train, data_val, epochs=500, lr=0.01, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data_train.x, data_train.edge_index).squeeze()
        loss = criterion(out, data_train.y)
        loss.backward()
        optimizer.step()

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_out = model(data_val.x, data_val.edge_index).squeeze()
            val_loss = criterion(val_out, data_val.y)
            val_preds = torch.sigmoid(val_out).cpu().numpy()
            val_labels = data_val.y.cpu().numpy()
            val_auc = roc_auc_score(val_labels, val_preds)

        #print(f"Epoch {epoch:03d}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val AUC: {val_auc:.4f}")

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
            
    model_path= os.path.join(save_dir, "GCN_best_model.pth")

    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), model_path)
     
    return model

