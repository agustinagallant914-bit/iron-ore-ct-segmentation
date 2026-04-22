import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import subprocess
import gc

# ========================================================
# 1. 全局设置
# ========================================================
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


# ========================================================
# 2. Dataset (共享)
# ========================================================
class OfflinePatchDataset(Dataset):
    def __init__(self, file_ids, data_dir, augment=True):
        self.file_ids = file_ids
        self.img_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.augment = augment

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        fid = self.file_ids[idx]
        try:
            # 读取 .npy (H,W,4) -> RGB + Gradient
            img = np.load(os.path.join(self.img_dir, f"{fid}.npy"))
            mask = cv2.imread(os.path.join(self.mask_dir, f"{fid}.png"), cv2.IMREAD_GRAYSCALE)

            if mask is None: raise ValueError("Mask is None")
            mask[mask >= 4] = 0

        except Exception as e:
            return torch.zeros(4, 256, 256), torch.zeros(256, 256).long()

        if self.augment:
            if np.random.random() > 0.5:
                img = np.flip(img, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            if np.random.random() > 0.5:
                img = np.flip(img, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).long()
        return img_t, mask_t


# ========================================================
# 3. 模型: Dual-Stream ResUNet
# ========================================================


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                  nn.BatchNorm2d(out_ch)) if in_ch != out_ch else nn.Identity()

    def forward(self, x): return F.relu(self.bn2(self.c2(F.relu(self.bn1(self.c1(x))))) + self.skip(x))


class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[1, 6, 12, 18]):
        super().__init__()
        self.aspp_blocks = nn.ModuleList()
        self.aspp_blocks.append(
            nn.Sequential(nn.Conv2d(in_dims, out_dims, 1, bias=False), nn.BatchNorm2d(out_dims), nn.ReLU(inplace=True)))
        for r in rate[1:]:
            self.aspp_blocks.append(nn.Sequential(nn.Conv2d(in_dims, out_dims, 3, padding=r, dilation=r, bias=False),
                                                  nn.BatchNorm2d(out_dims), nn.ReLU(inplace=True)))
        self.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_dims, out_dims, 1, bias=False),
                                         nn.BatchNorm2d(out_dims), nn.ReLU(inplace=True))
        self.out_conv = nn.Sequential(nn.Conv2d(out_dims * (len(rate) + 1), out_dims, 1, bias=False),
                                      nn.BatchNorm2d(out_dims), nn.ReLU(inplace=True))

    def forward(self, x):
        x_pool = F.interpolate(self.global_pool(x), size=x.shape[2:], mode='bilinear', align_corners=True)
        out = [block(x) for block in self.aspp_blocks] + [x_pool]
        return self.out_conv(torch.cat(out, dim=1))


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, padding=0, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, padding=0, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, padding=0, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x): return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))


class EdgeGate(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, main_feat, edge_feat):
        return main_feat + (main_feat * self.sigmoid(self.conv(edge_feat)))


class DualResUNet(nn.Module):
    def __init__(self, n_cl=4):
        super().__init__()
        self.rgb_conv = nn.Conv2d(3, 64, 3, padding=1)
        self.e1 = ResBlock(64, 64)
        self.e2 = ResBlock(64, 128)
        self.e3 = ResBlock(128, 256)
        self.e4 = ResBlock(256, 512)

        self.grad_conv = nn.Conv2d(1, 32, 3, padding=1)
        self.edge_e1 = nn.Conv2d(32, 64, 3, padding=1)
        self.edge_e2 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.edge_e3 = nn.Conv2d(128, 256, 3, padding=1, stride=2)

        self.gate1, self.gate2, self.gate3 = EdgeGate(64), EdgeGate(128), EdgeGate(256)

        self.bt = ResBlock(512, 1024)
        self.aspp = ASPP(1024, 1024)

        # Up-sampling parameters directly defined here
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.ag4 = AttentionGate(512, 512, 256)
        self.d4 = ResBlock(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.ag3 = AttentionGate(256, 256, 128)
        self.d3 = ResBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.ag2 = AttentionGate(128, 128, 64)
        self.d2 = ResBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.ag1 = AttentionGate(64, 64, 32)
        self.d1 = ResBlock(128, 64)

        self.out = nn.Conv2d(64, n_cl, 1)

    def forward(self, x):
        rgb, grad = x[:, :3, :, :], x[:, 3:, :, :]

        eg0 = F.relu(self.grad_conv(grad))
        eg1 = self.edge_e1(eg0)
        eg2 = self.edge_e2(eg1)
        eg3 = self.edge_e3(eg2)

        x0 = F.relu(self.rgb_conv(rgb))
        x1 = self.gate1(self.e1(x0), eg1)
        x2 = self.gate2(self.e2(F.max_pool2d(x1, 2)), eg2)
        x3 = self.gate3(self.e3(F.max_pool2d(x2, 2)), eg3)
        x4 = self.e4(F.max_pool2d(x3, 2))

        b = self.aspp(self.bt(F.max_pool2d(x4, 2)))
        d4_up = self.up4(b)
        d4 = self.d4(torch.cat([self.ag4(d4_up, x4), d4_up], 1))
        d3_up = self.up3(d4)
        d3 = self.d3(torch.cat([self.ag3(d3_up, x3), d3_up], 1))
        d2_up = self.up2(d3)
        d2 = self.d2(torch.cat([self.ag2(d2_up, x2), d2_up], 1))
        d1_up = self.up1(d2)
        d1 = self.d1(torch.cat([self.ag1(d1_up, x1), d1_up], 1))
        return self.out(d1)


# ========================================================
# 4. 通用评估与训练函数
# ========================================================
def calculate_metrics(cm):
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    eps = 1e-7
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * (prec * rec) / (prec + rec + eps)
    return iou, dice, prec, rec, f1


def train_and_evaluate(model_name, model, train_loader, val_loader, test_loader, epochs=100):
    save_dir = f"U_Result_{model_name}"
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    print(f"\n{'=' * 20} Start Training: {model_name} {'=' * 20}")
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 2.0, 2.0, 2.0]).to(device))

    best_iou = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = criterion(model(imgs), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        v_cm = np.zeros((4, 4), dtype=np.int64)
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                out = model(imgs)
                preds = torch.argmax(out, 1).cpu().numpy().flatten()
                labels = masks.cpu().numpy().flatten()
                valid = labels < 4
                if valid.any():
                    v_cm += confusion_matrix(labels[valid], preds[valid], labels=[0, 1, 2, 3])

        iou, _, _, _, _ = calculate_metrics(v_cm)
        mIoU = np.mean(iou[1:])  # Skip background

        print(f"[{model_name}] Epoch {epoch + 1} | Loss: {train_loss / len(train_loader):.4f} | mIoU: {mIoU:.4f}")

        if mIoU > best_iou:
            best_iou = mIoU
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

    # Final Test
    print(f"\n>>> Testing Best Model for {model_name}...")
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth"), weights_only=True))
    model.eval()
    cm = np.zeros((4, 4), dtype=np.int64)
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)
            preds = torch.argmax(out, 1).cpu().numpy().flatten()
            labels = masks.cpu().numpy().flatten()
            valid = labels < 4
            if valid.any():
                cm += confusion_matrix(labels[valid], preds[valid], labels=[0, 1, 2, 3])

    iou, dice, prec, rec, f1 = calculate_metrics(cm)
    classes = ["Red", "Green", "Blue"]
    print(f"\nFinal Results for {model_name}:")
    print(f"{'Class':<10} {'IoU':<8} {'Dice':<8} {'Precision':<10} {'Recall':<8} {'F1':<8}")
    for i in range(3):
        idx = i + 1
        print(f"{classes[i]:<10} {iou[idx]:.4f}   {dice[idx]:.4f}   {prec[idx]:.4f}    {rec[idx]:.4f}   {f1[idx]:.4f}")
    print("-" * 60)
    print(
        f"{'Mean':<10} {np.mean(iou[1:]):.4f}   {np.mean(dice[1:]):.4f}   {np.mean(prec[1:]):.4f}    {np.mean(rec[1:]):.4f}   {np.mean(f1[1:]):.4f}")
    print("=" * 60 + "\n")

    # Clean up memory
    del model
    del optimizer
    torch.cuda.empty_cache()
    gc.collect()


# ========================================================
# 5. 主程序入口
# ========================================================
def main():
    data_dir = "local_patches"
    if not os.path.exists(os.path.join(data_dir, "images")):
        print("Data not found.")
        return

    all_fids = [f.split('.')[0] for f in os.listdir(os.path.join(data_dir, "images")) if f.endswith('.npy')]
    if not all_fids: return

    train_fids, temp = train_test_split(all_fids, test_size=0.2, random_state=42)
    val_fids, test_fids = train_test_split(temp, test_size=0.5, random_state=42)

    # Dataloaders
    train_loader = DataLoader(OfflinePatchDataset(train_fids, data_dir, True), batch_size=4, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(OfflinePatchDataset(val_fids, data_dir, False), batch_size=4, shuffle=False, num_workers=2)
    test_loader = DataLoader(OfflinePatchDataset(test_fids, data_dir, False), batch_size=4, shuffle=False,
                             num_workers=2)

    # --- 实验: 双流 ResUNet (Ours) ---
    # Dual stream input (Explicit Decoupling)
    model_ours = DualResUNet(n_cl=4).to(device)
    train_and_evaluate("Ours_DualResUNet", model_ours, train_loader, val_loader, test_loader)


if __name__ == "__main__":
    main()