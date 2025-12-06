from model.model_race_winner import RaceWinnerPredictor
import pandas as pd
import numpy as np
import optuna
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

PROCESSED_DIR = Path('data/processed')

DB_DIR = Path('optuna')
DB_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

race_df = pd.read_parquet(PROCESSED_DIR / "race.parquet")
horse_df = pd.read_parquet(PROCESSED_DIR / "horse.parquet")
race_cols = [c for c in race_df.columns if c not in ["race_id", "horse_id"]]
horse_cols = [c for c in horse_df.columns if c not in ["is_win", "is_show", "win_odds", "race_id"]]


def tune_by_optuna():
    study_name = "torch_study"
    db_path = f"sqlite:///{DB_DIR}/{study_name}_optuna.db"
    study = optuna.create_study(
        direction='minimize', study_name=study_name, storage=db_path, load_if_exists=True
    )
    study.optimize(objective, n_trials=50)
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params:")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    return trial


def build_cfg(trial, suffix, act):
    cfg =  {
        "hidden": trial.suggest_int(f"{suffix}_hidden", 64, 256, step=64),
        "n_layers": trial.suggest_int(f"{suffix}_n_layers", 3, 6),
        "dropout": trial.suggest_float(f"{suffix}_dropout", 0.0, 0.3),
        "act": act
    }
    if suffix == "attn":
        cfg["n_heads"] = trial.suggest_int(f"attn_n_heads", 2, 8)
    return cfg




def objective(trial):
    mlp_act = trial.suggest_categorical("mlp_act", ["gelu", "silu"])
    attn_act = trial.suggest_categorical("attn_act", ["relu", "gelu"])
    horse_cfg = build_cfg(trial, "horse", mlp_act)
    race_cfg = build_cfg(trial, "race", mlp_act)
    attn_cfg = build_cfg(trial, "attn", attn_act)
    head_cfg = build_cfg(trial, "head", mlp_act)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    max_epochs = trial.suggest_int("epochs", 10, 30)
    batch_size = trial.suggest_categorical("batch_size", [128, 256])
    
    gkf = GroupKFold(n_splits=3)
    fold_scores = []
    ce_means, bce_means = [], []
    top1_acc, top3_acc = [], [] 
    # 3️⃣ Training loop (simplified)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(race_df, groups=race_df["race_id"])):
        train_race_df = race_df.iloc[train_idx]
        val_race_df = race_df.iloc[val_idx]

        train_ds = RaceGroupDataset(horse_df, train_race_df, horse_cols, race_cols)
        val_ds   = RaceGroupDataset(horse_df, val_race_df, horse_cols, race_cols)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_races)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_races)

        model = RaceWinnerPredictor(
            horse_dim=len(horse_cols), 
            race_dim=len(race_cols),
            horse_cfg=horse_cfg, 
            race_cfg=race_cfg, 
            attn_cfg=attn_cfg, 
            head_cfg=head_cfg
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        for epoch in range(max_epochs):  # few epochs for tuning
            model.train()
            for horse_feats, race_feats, labels, _, mask in train_loader:
                horse_feats, race_feats, labels, mask = [
                    x.to(device) for x in [horse_feats, race_feats, labels, mask]
                ]
                optimizer.zero_grad()
                logits, _ = model(horse_feats, race_feats, mask)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
        # 4️⃣ Evaluation
        model.eval()
        ce_losses, bce_losses = [], []
        correct_top1, correct_top3, total = 0, 0, 0

        with torch.no_grad():
            for horse_feats, race_feats, win_targets, show_targets, mask in val_loader:
                horse_feats, race_feats, win_targets, show_targets, mask = (
                    horse_feats.to(device),
                    race_feats.to(device),
                    win_targets.to(device),
                    show_targets.to(device),
                    mask.to(device)
                )

                logits, _ = model(horse_feats, race_feats, mask)
                logits = logits.masked_fill(~mask, -1e9)

                # --- CE (top-1) loss ---
                ce_loss = F.cross_entropy(logits, win_targets)
                ce_losses.append(ce_loss.item())

                # --- BCE (top-3) eval metric ---
                probs = F.softmax(logits, dim=1)
                bce_loss = F.binary_cross_entropy(probs[mask].flatten(), show_targets[mask].flatten())
                bce_losses.append(bce_loss.item())

                # --- top-k metrics ---
                pred_topk = probs.topk(3, dim=1).indices
                correct_top1 += (pred_topk[:, 0] == win_targets).sum().item()
                correct_top3 += (pred_topk == win_targets.unsqueeze(1)).any(1).sum().item()
                total += len(win_targets)

            ce_mean = sum(ce_losses) / len(ce_losses)
            bce_mean = sum(bce_losses) / len(bce_losses)
            fold_score = 0.7 * ce_mean + 0.3 * bce_mean
            ce_means.append(ce_mean)
            bce_means.append(bce_mean)
            top1_acc.append(correct_top1 / total)
            top3_acc.append(correct_top3 / total)
            fold_scores.append(fold_score)
        
    trial.set_user_attr("win_ce_loss", np.mean(ce_means))
    trial.set_user_attr("show_bce_loss", np.mean(bce_means))
    trial.set_user_attr("top1_acc", np.mean(top1_acc))
    trial.set_user_attr("top3_acc", np.mean(top3_acc))

    cfg = {"horse": horse_cfg, "race": race_cfg, "attn": attn_cfg, "head": head_cfg}
    trial.set_user_attr("cfg", cfg)

    return np.mean(fold_scores)


    
