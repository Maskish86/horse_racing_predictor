
class RaceGroupDataset(Dataset):
    def __init__(self, horse_df, race_df, horse_cols, race_cols):
        self.horses_df = horse_df
        self.race_df = race_df
        self.horse_cols = horse_cols
        self.race_cols = race_cols
        self.race_ids = race_df["race_id"].unique()

    def __len__(self):
        return len(self.race_ids)

    def __getitem__(self, idx):
        race_id = self.race_ids[idx]
        horses = self.horses_df[self.horses_df["race_id"] == race_id]

        horse_feats = horses[self.horse_cols].values.astype("float32")
        race_feats = self.race_df[self.race_df["race_id"] == race_id][self.race_cols].values[0].astype("float32")

        win_idx = horses["is_win"].values.argmax()  # for CE
        show_flags = horses["is_show"].values.astype("float32")  # for BCE evaluation

        return torch.tensor(horse_feats), torch.tensor(race_feats), win_idx, torch.tensor(show_flags)


def collate_races(batch):
    horse_list, race_list, win_idx_list, show_list = zip(*batch)
    max_horses = max(h.shape[0] for h in horse_list)

    padded_horses, padded_shows, masks = [], [], []
    for horses, show in zip(horse_list, show_list):
        n = horses.shape[0]
        pad = max_horses - n

        # Pad features
        padded_horses.append(torch.cat([horses, torch.zeros(pad, horses.shape[1])], dim=0))

        # Pad is_show
        padded_shows.append(torch.cat([show, torch.zeros(pad)], dim=0))

        # Mask
        mask = torch.zeros(max_horses, dtype=torch.bool)
        mask[:n] = True
        masks.append(mask)

    horse_feats = torch.stack(padded_horses)          # [batch, max_horses, horse_dim]
    race_feats  = torch.stack(race_list)              # [batch, race_dim]
    mask        = torch.stack(masks)                  # [batch, max_horses]
    win_targets = torch.tensor(win_idx_list, dtype=torch.long)  # [batch]
    show_targets = torch.stack(padded_shows)          # [batch, max_horses]

    return horse_feats, race_feats, win_targets, show_targets, mask


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def step(self, val_loss):
        if self.best_score is None or val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
