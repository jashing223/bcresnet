import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# 引入你的專案模組
from bcresnet import BCResNets
from utils import SpeechCommand, Padding, Preprocess

def plot_confusion_matrix(ckpt_path, gpu_id=0):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {os.path.basename(ckpt_path)}")
    
    # 1. 載入模型
    model = BCResNets(base_c=int(8.0 * 8)).to(device)
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    model.eval()
    
    # 2. 載入驗證集
    base_dir = "/home/jashing223/datasets/GSC/speech_commands_v0.02" 
    valid_dir = os.path.join(base_dir, "valid_12class")
    embeddings_path = "./valid_embeddings.pt"
    
    transform = transforms.Compose([Padding()])
    dataset = SpeechCommand(valid_dir, ver=2, transform=transform, embeddings_path=embeddings_path)
    loader = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=False)
    preprocess = Preprocess(None, device)
    
    # GSC V2 Labels
    class_names = ['_silence_', '_unknown_', 'down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
    
    all_preds = []
    all_labels = []
    
    print("Generating Confusion Matrix...")
    with torch.no_grad():
        for sample in tqdm(loader):
            inputs, embeddings, labels, _ = sample
            inputs = inputs.to(device)
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            inputs = preprocess(inputs, labels=labels, augment=False, is_train=False)
            
            # 取得預測
            # 我們主要看 classifier3 (Multiclass) 的表現，因為這決定了最終的 ACC
            _, _, keyword_class_logits = model(inputs, embeddings)
            
            # Logic:
            # 1. 如果是 Silence (Label 0) -> 應該對應到 Silence
            # 2. 如果是 Unknown (Label 1) -> 應該對應到 Unknown
            # 3. 其他 -> 對應到具體 class
            
            # 由於你的 classifier3 輸出通常是 10 類 (不含 sil/unk) 或 12 類?
            # 根據 bcresnet.py: self.classifier3 輸出是 num_classes-2 (即 10 類)
            # 所以我們需要結合 classifier2 (Binary) 來判斷 Unk/Sil
            
            # 這裡我們模擬 main.py 的邏輯：
            # 如果 classifier2 說 "不是 keyword" (prob < 0.5) -> 判斷為 Unknown (或是 Silence，這邊簡化為 Unknown)
            # 如果 classifier2 說 "是 keyword" -> 看 classifier3 說是哪一個
            
            # 為了診斷"ACC上不去"的原因，我們直接看 classifier3 對 12 類的掌握度
            # 但因為結構限制，我們先用最單純的方式：看 classifier3 在 keyword 上的表現
            # 以及 classifier2 在 unknown 上的表現
            
            # --- 修正診斷邏輯 ---
            # 我們手動拼湊出 12 類的預測結果
            # 0: Silence, 1: Unknown, 2~11: Keywords
            
            # 1. 取得 classifier2 (Binary) 分數
            _, keyword_logits, _ = model(inputs, embeddings)
            prob_is_keyword = torch.sigmoid(keyword_logits) # [B, 1]
            
            # 2. 取得 classifier3 (Class) 分數
            probs_class = F.softmax(keyword_class_logits, dim=1) # [B, 10]
            pred_class_idx = torch.argmax(probs_class, dim=1) + 2 # +2 因為 0,1 被占用了
            
            batch_preds = []
            for i in range(len(labels)):
                # 如果是 Silence (這裡 dataset 應該很少 silence，主要看 unknown)
                if labels[i] == 0: 
                    batch_preds.append(0) # 假設 silence 總是被正確過濾 (通常依靠 VAD，這邊簡化)
                    continue

                # 判斷邏輯：
                # 如果 classifier2 信心度 < 0.5 -> 預測為 Unknown (1)
                # 否則 -> 預測為 classifier3 的結果
                if prob_is_keyword[i] < 0.5:
                    batch_preds.append(1)
                else:
                    batch_preds.append(pred_class_idx[i].item())
            
            all_preds.extend(batch_preds)
            all_labels.extend(labels.cpu().numpy())

    # 計算混淆矩陣
    cm = confusion_matrix(all_labels, all_preds, labels=range(12))
    
    # 計算每個類別的 Accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    print("\n[類別準確率診斷]")
    for i, name in enumerate(class_names):
        # 避免除以 0
        count = cm.sum(axis=1)[i]
        if count > 0:
            print(f"{name:10s}: {per_class_acc[i]*100:6.2f}% (樣本數: {count})")
        else:
            print(f"{name:10s}: N/A")

    # 繪圖
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Validation Set)')
    plt.savefig('confusion_matrix.png')
    print("\n混淆矩陣已儲存為 confusion_matrix.png")

    # 關鍵診斷
    unk_as_kw = cm[1, 2:].sum() # 真值是 Unknown，被預測為 Keyword (2~11)
    unk_total = cm[1].sum()
    print(f"\n[關鍵問題分析]")
    print(f"Unknown 被誤判為 Keyword (FA 來源): {unk_as_kw} / {unk_total} ({unk_as_kw/unk_total*100:.2f}%)")
    
    kw_as_unk = cm[2:, 1].sum() # 真值是 Keyword，被預測為 Unknown (Miss)
    kw_total = cm[2:].sum() # 所有 Keyword 樣本數 (不精確，但夠用)
    print(f"Keyword 被誤判為 Unknown (Recall 損失): {kw_as_unk} / {kw_total} ({kw_as_unk/kw_total*100:.2f}%)")

if __name__ == "__main__":
    # 替換成你第 10 epoch 的 checkpoint
    CKPT = "./checkpoints/pvad_sr_tau_8.0_ver_2_ERes2NetV2/model_epoch_11_acc_67.78.ckpt" 
    plot_confusion_matrix(CKPT)