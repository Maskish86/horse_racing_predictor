from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle
from sklearn.metrics import log_loss, mean_absolute_error, brier_score_loss
import optuna
import copy
import gc
import matplotlib.pyplot as plt

MODEL_DIR = Path('models/mlp_roi')
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR = Path('optuna/mlp_roi')
DB_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Using GPU:', torch.cuda.get_device_name(0))

class RaceNet(nn.Module):
    def __init__(self, input_dim, num_layers, units, dropout_rate, decay_rate, cat_dims, emb_dim=8):
        super().__init__()

        self.emb_layers = nn.ModuleList([
            nn.Embedding(num_categories, emb_dim) for num_categories in cat_dims
        ])
        total_emb_dim = emb_dim * len(cat_dims)

        layers = []
        for i in range(num_layers):
            out_dim = int(units * (decay_rate ** i))
            if out_dim < 8:
                break
            input_size = input_dim + total_emb_dim if i == 0 else prev_dim
            layers.append(nn.Linear(input_size, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = out_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x_cat, x_cont):
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.emb_layers)]
        x_emb = torch.cat(embedded, dim=1)

        x = torch.cat([x_emb, x_cont], dim=1)
        return self.model(x).view(-1)

class EarlyStopping:
    def __init__(self, mode='max', patience=5, verbose=False, min_delta=1e-4):
        assert mode in ['min', 'max'], 'mode must be 'min' or 'max''
        self.mode = mode
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta

        self.best_score = None
        self.best_model_state = None
        self.wait = 0
        self.stopped = False
        self.eval_step = 0
        self.best_step = 0

    def __call__(self, score, model):
        self.eval_step += 1  # Called every 5 epochs, not every epoch

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            return False

        improve = (score > self.best_score + self.min_delta) if self.mode == 'max' else (score < self.best_score - self.min_delta)
        if improve:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.wait = 0
            self.best_step = self.eval_step
            if self.verbose:
                print(f'[EarlyStopping] Step {self.eval_step}: Improved {self.mode} metric to {score:.5f}')
        else:
            self.wait += 1
            if self.verbose:
                print(f'[EarlyStopping] Step {self.eval_step}: No improvement. Patience {self.wait}/{self.patience}')
            if self.wait >= self.patience:
                self.stopped = True
        return self.stopped

    def get_best_state(self):
        return self.best_model_state

    def best_was_recent(self):
        return (self.eval_step - self.best_step) <= self.patience

def train_model_with_early_stopping(model,
                                    X_train_cont, X_train_cat, y_train,
                                    X_valid_cont, X_valid_cat, y_valid,
                                    race_valid, win_odds_valid,
                                    share_threshold, ev_threshold,
                                    batch_size, learning_rate,
                                    device,
                                    max_epochs=150, check_interval=5, verbose=False):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
                                              steps_per_epoch=int(np.ceil(len(X_train_cont) / batch_size)),
                                              epochs=max_epochs)
    criterion = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(mode='max', patience=5)

    train_loader = DataLoader(TensorDataset(X_train_cont, X_train_cat, y_train),
                              batch_size=batch_size, shuffle=True)

    for epoch in range(max_epochs):
        model.train()
        for xb_cont, xb_cat, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb_cat, xb_cont)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()

        if epoch % check_interval == 0:
            model.eval()
            with torch.no_grad():
                val_preds = torch.sigmoid(model(X_valid_cat, X_valid_cont)).cpu().numpy()

            df_valid = pd.DataFrame({
                'race_id': race_valid,
                'win_odds': win_odds_valid,
                'pred_proba': val_preds.flatten(),
                'true': y_valid.cpu().numpy()
            })
            df_valid['proba_share'] = df_valid['pred_proba'] / df_valid.groupby('race_id', observed=True)['pred_proba'].transform('sum')
            df_valid['expected_value'] = df_valid['win_odds'] * df_valid['proba_share']
            df_valid['pred'] = ((1 / df_valid['proba_share']) < share_threshold) & \
                               (df_valid['expected_value'] > ev_threshold)
            df_pred = df_valid[df_valid['pred']]
            total_reward = (df_pred['win_odds'] * df_pred['true']).sum()
            total_cost = df_pred['win_odds'].sum()
            val_roi = total_reward / total_cost if total_cost > 0 else 0.0

            if verbose:
                print(f'Epoch {epoch} ROI: {val_roi:.5f}')
            if early_stopping(val_roi, model):
                print(f'Early stopping at epoch {epoch}')
                break

    best_state = early_stopping.get_best_state()
    if best_state:
        model.load_state_dict(best_state)

    return model, early_stopping.best_score


def prepare_data(df, cont_cols, cat_cols, scaler=None, fit_scaler=True):
    X_cont_np = df[cont_cols].values
    if fit_scaler or scaler is None:
        scaler = StandardScaler().fit(X_cont_np)
    X_cont = torch.tensor(scaler.transform(X_cont_np), dtype=torch.float32).to(DEVICE)

    X_cat = df[cat_cols].copy()
    cat_dims = []
    for col in cat_cols:
        X_cat[col], uniques = pd.factorize(X_cat[col])
        cat_dims.append(len(uniques))
    X_cat = torch.tensor(X_cat.values, dtype=torch.long).to(DEVICE)

    return X_cont, X_cat, cat_dims, scaler


def train_pytorch_by_roi(target_name, trials):
    db_path = f'sqlite:///{DB_DIR}/optuna_{target_name}.db'
    final_df = pd.read_csv('/content/drive/MyDrive/horse_racing_predictor/csv/final_df.csv', encoding='utf-8')
    print(final_df.head(5).to_string())

    train_df = final_df[(final_df['race_id'] > 201200000000) & (final_df['race_id'] < 202400000000)].reset_index(drop=True)
    train_df = train_df[train_df.groupby('race_id')['is_insufficient_history-5'].transform(lambda x: not all(x == 1))].reset_index(drop=True)
    train_df = train_df.replace(np.inf, 0)
    print(train_df.shape)
    test_df = final_df[(final_df['race_id'] > 202400000000) & (final_df['race_id'] < 202500000000)].reset_index(drop=True)
    del final_df
    gc.collect()

    drop_cols = ['is_win', 'is_place', 'is_show', 'race_id', 'datetime', 'win_odds', 'horse_number', 'location']
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    cat_cols = [col for col in feature_cols if 'cat' in col]
    cont_cols = [col for col in feature_cols if col not in cat_cols]

    tune_df = train_df[train_df['race_id'] >= 202100000000]
    race_id_array_tune = tune_df['race_id'].values
    unique_groups = np.unique(race_id_array_tune)
    shuffled_groups = shuffle(unique_groups, random_state=42)
    tune_df = tune_df[tune_df['race_id'].isin(shuffled_groups)]
    tune_df['race_id'] = pd.Categorical(tune_df['race_id'], categories=shuffled_groups, ordered=True)
    tune_df = tune_df.sort_values('race_id').reset_index(drop=True)
    print(tune_df.shape)

    X_cont_tune, X_cat_tune, cat_dims, scaler = prepare_data(tune_df, cont_cols, cat_cols)
    y_tensor_tune = torch.tensor(tune_df[target_name].values, dtype=torch.float32).to(DEVICE)
    race_id_array_tune = tune_df['race_id'].values
    win_odds_array_tune = tune_df['win_odds'].values
    del tune_df
    gc.collect()

    study = optuna.create_study(direction='minimize', study_name=f'torch_{target_name}_opt_roi', storage=db_path, load_if_exists=True)
    study.optimize(pytorch_objective, n_trials=trials)

    def pytorch_objective(trial):
        num_layers = trial.suggest_int('num_layers', 6, 8)
        units = trial.suggest_int('units', 128, 1024, step=128)
        dropout_rate = trial.suggest_float('dropout_rate', 0.6, 0.9)
        learning_rate = trial.suggest_float('learning_rate', 5e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
        decay_rate = trial.suggest_float('decay_rate', 0.4, 0.8)
        emb_dim = trial.suggest_int('emb_dim', 8, 32)
        share_threshold = trial.suggest_float('share_threshold', 1.5, 4, step=0.1)
        ev_threshold = trial.suggest_float('ev_threshold', 2, 5, step=0.1)

        group_kfold = GroupKFold(n_splits=3)
        fold_scores = []


        for fold_idx, (train_idx, valid_idx) in enumerate(group_kfold.split(X_cont, y_tensor, groups=race_id_array)):
            model = RaceNet(X_cont.shape[1], num_layers, units, dropout_rate, decay_rate, cat_dims, emb_dim).to(DEVICE)
            X_train_cont, X_train_cat = X_cont[train_idx], X_cat[train_idx]
            y_train = y_tensor[train_idx]
            X_valid_cont, X_valid_cat = X_cont[valid_idx], X_cat[valid_idx]
            y_valid = y_tensor[valid_idx]
            race_valid = race_id_array[valid_idx]
            win_odds_valid = win_odds_array[valid_idx]

            model, roi_score = train_model_with_early_stopping(
                model,
                X_train_cont, X_train_cat, y_train,
                X_valid_cont, X_valid_cat, y_valid,
                race_valid, win_odds_valid,
                share_threshold, ev_threshold,
                batch_size, learning_rate,
                DEVICE
            )
            fold_scores.append(roi_score)
            torch.cuda.empty_cache()

        return -np.mean(fold_scores)

    

    del X_cont_tune, X_cat_tune, cat_dims, y_tensor_tune, race_id_array_tune, win_odds_array_tune
    gc.collect()

    best_params = study.best_params
    print('Best Hyperparameters:', best_params)
    train_df = train_df.sort_values('race_id', ascending=True).reset_index(drop=True)
    X_cont, X_cat, cat_dims, scaler = prepare_data(train_df, cont_cols, cat_cols, scaler)
    y_tensor = torch.tensor(train_df[target_name].values, dtype=torch.float32).to(DEVICE)
    race_id_array = train_df['race_id'].values
    win_odds_array = train_df['win_odds'].values
    del train_df
    gc.collect()

    model = RaceNet(
        input_dim=X_cont.shape[1],
        num_layers=best_params['num_layers'],
        units=best_params['units'],
        dropout_rate=best_params['dropout_rate'],
        decay_rate=best_params['decay_rate'],
        cat_dims=cat_dims,
        emb_dim=best_params['emb_dim']
    ).to(DEVICE)

    gkf = GroupKFold(n_splits=3)
    splits = list(gkf.split(X_cont, y_tensor, groups=race_id_array))
    train_idx, valid_idx = splits[-1]

    X_train_cont, X_train_cat = X_cont[train_idx], X_cat[train_idx]
    y_train = torch.tensor(y_tensor[train_idx], dtype=torch.float32)
    X_valid_cont, X_valid_cat = X_cont[valid_idx], X_cat[valid_idx]
    y_valid = torch.tensor(y_tensor[valid_idx], dtype=torch.float32)
    race_valid = race_id_array[valid_idx]
    win_odds_valid = win_odds_array[valid_idx]

    model, final_roi = train_model_with_early_stopping(
        model,
        X_train_cont, X_train_cat, y_train,
        X_valid_cont, X_valid_cat, y_valid,
        race_valid, win_odds_valid,
        best_params['share_threshold'], best_params['ev_threshold'],
        best_params['batch_size'], best_params['learning_rate'],
        DEVICE,
        verbose=True
    )
    print(f'Best ROI on validation: {final_roi:.4f}')

    torch.save({
        'model_state_dict': model.state_dict(),
        'best_params': best_params,
        'cat_dims': cat_dims
    }, MODEL_DIR / f'torch_{target_name}_roi_model.pth')

    model.eval()
    pred_best = []
    with torch.no_grad():
        for i in range(0, len(X_cat), 2048):
            batch_cat = X_cat[i:i + 2048]
            batch_cont = X_cont[i:i + 2048]
            batch_pred = torch.sigmoid(model(batch_cat, batch_cont)).cpu().numpy()
            pred_best.append(batch_pred)

    pred_best = np.concatenate(pred_best).flatten().clip(0.001, 0.999)

    print(f'Log Loss: {log_loss(y_tensor, pred_best)}')
    print(f'MAE: {mean_absolute_error(y_tensor, pred_best)}')
    print(f'Brier Score: {brier_score_loss(y_tensor, pred_best)}')

    return model
