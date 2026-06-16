from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np


class SubsetDataset(Dataset):
    def __init__(self, seq_path, track_path, annotations_path, case_identifier, transform=None):
        X_seq, X_track, y_matched, prefix_tid = select_specific_cases(seq_path, track_path, annotations_path, case_identifier)

        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_track = torch.tensor(X_track, dtype=torch.float32)
        self.prefix_tid = prefix_tid
        self.transform = transform  

    def __len__(self):
        # Total number of samples
        return len(self.prefix_tid)

    def __getitem__(self, idx):
        seq = self.X_seq[idx]
        track = self.X_track[idx]
        prefix_tid = self.prefix_tid[idx]

        # Apply optional transform to features
        if self.transform:
            seq, track = self.transform((seq, track))

        return seq, track, prefix_tid 


def select_specific_cases(seq_path, track_path, annotations_path, case_identifier):
    specfic_cases = []
    annotations_df = pd.read_excel(annotations_path)
    specfic_cases = annotations_df.loc[
        annotations_df["Test Set"] == case_identifier, "Case Name"
    ].tolist()

    seq_data = np.load(seq_path, allow_pickle=True)
    track_data = np.load(track_path, allow_pickle=True)

    X_seq, y_seq, track_ids_seq = seq_data['X'], seq_data['y'], seq_data['track_ids']
    X_track, y_track, track_ids_track = track_data['X'], track_data['y'], track_data['track_ids']

    if X_seq.shape[1] == 11 and X_seq.shape[2] == 20:
        print("transposing...")
        X_seq = np.transpose(X_seq, (0, 2, 1))

    track_id_to_index = {
        tuple(tid) if isinstance(tid, (list, tuple, np.ndarray)) else (tid,): i
        for i, tid in enumerate(track_ids_track)
    }

    X_seq_matched, X_track_matched, y_matched, prefix_tid = [], [], [], []
    for i, tid in enumerate(track_ids_seq):
        key = tuple(tid) if isinstance(tid, (list, tuple, np.ndarray)) else (tid,)
        if key in track_id_to_index:
            prefix_split = "_".join(tid[0].split("_")[:2])
            if prefix_split in specfic_cases:
                idx = track_id_to_index[key]
                X_seq_matched.append(X_seq[i])
                X_track_matched.append(X_track[idx])
                y_matched.append(y_seq[i])
                prefix_tid.append(tid[0]+str(tid[1]))

    return X_seq_matched, X_track_matched, y_matched, prefix_tid