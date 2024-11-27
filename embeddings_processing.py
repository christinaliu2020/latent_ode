import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lib.utils as utils
import torch.nn.functional as F

from sklearn.decomposition import PCA


class MouseVideoEmbeddings(Dataset):
    def __init__(self, embeddings_file, labels_file, keypoints_file = None, sequence_length = 50, keypoint_window = 100, overlap=0, max_seq_length=None, split_sequences=False, do_pca = False, normalize = True,num_splits=20, device=torch.device("cpu")):
        self.embeddings = []
        self.labels = []

        for emb_file, label_file in zip(embeddings_file, labels_file):
            emb = torch.Tensor(np.load(emb_file)).to(device)
            self.embeddings.append(emb)
            self.labels.extend(np.load(label_file))

        self.embeddings = torch.cat(self.embeddings, dim=0)
        self.labels = np.array(self.labels)
        if keypoints_file:
            self.keypoints = torch.Tensor(np.load(keypoints_file)).to(device)
        else:
            self.keypoints = None
        self.keypoint_window = keypoint_window


        self.max_seq_length = max_seq_length
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.split_sequences = split_sequences
        self.num_splits = num_splits
        self.num_classes = self.get_num_classes()

        if normalize:

            self.embeddings, self.data_min, self.data_max = utils.normalize_data(self.embeddings)

        self.embeddings = self.embeddings.cpu().numpy()

        self.frame_ids = np.arange(len(self.embeddings))
        if do_pca:
            pca = PCA(n_components=64)
            self.embeddings = pca.fit_transform(self.embeddings)

        self.labels = self.encode_labels(self.labels)
        if split_sequences:
            if self.keypoints:
                self.split_data_keypoints()
            else:
                self.split_data()
        else:
            self.data = [self.embeddings]

    def normalize_embeddings(self):
        # Standardization normalization

        data = self.embeddings
        reshaped = data.reshape(-1, data.size(-1))

        att_mean = torch.mean(reshaped, 0)
        att_std = torch.std(reshaped, 0)

        # Avoid division by zero
        att_std[att_std == 0.] = 1.

        data_standardized = (data - att_mean) / att_std

        if torch.isnan(data_standardized).any():
            raise Exception("nans!")

        self.embeddings = data_standardized
        #return data_standardized, att_mean, att_std

    def split_data(self):
        total_frames = self.embeddings.shape[0]
        step = self.sequence_length #- self.overlap

        self.data = []
        for i in range(0, total_frames - self.sequence_length + 1, step):
            sequence = self.embeddings[i:i + self.sequence_length]
            labels = self.labels[i:i + self.sequence_length]
            frame_ids = np.arange(i, i+ self.sequence_length)
            self.data.append((sequence, labels, frame_ids))

    def split_data_keypoints(self):
        total_frames = self.embeddings.shape[0]
        step = self.sequence_length #- self.overlap
        keypoint_half_window = self.keypoint_window // 2
        start_frame = keypoint_half_window
        end_frame = total_frames - keypoint_half_window - self.sequence_length + 1


        self.data = []
        #for i in range(0, total_frames - self.sequence_length + 1, step):
        for i in range(start_frame, end_frame, self.sequence_length):
            sequence = self.embeddings[i:i + self.sequence_length]
            labels = self.labels[i:i + self.sequence_length]
            frame_ids = np.arange(i, i+ self.sequence_length)


            keypoints_sequence = []
            for j in range(i, i + self.sequence_length):
                #start = max(0, j - self.keypoint_window // 2)
                #end = min(total_frames, j + self.keypoint_window // 2 + 1)
                start = j - keypoint_half_window
                end = j + keypoint_half_window + 1
                frame_keypoints = self.keypoints[start:end].cpu().numpy()
                # pad_length = self.keypoint_window - len(frame_keypoints)
                # if pad_length > 0:
                #     frame_keypoints = np.pad(frame_keypoints, ((0, pad_length), (0, 0)), mode='constant')

                keypoints_sequence.append(frame_keypoints)
            keypoints_sequence = np.array(keypoints_sequence)
            self.data.append((sequence, labels, frame_ids, keypoints_sequence))


    def get_num_classes(self):
        return len(np.unique(self.labels))

    def __len__(self):
        return len(self.data)

    # def encode_labels(self, labels):
    #     num_classes = self.get_num_classes()
    #     one_hot = np.zeros((len(labels), num_classes))
    #     for i, label in enumerate(labels):
    #         one_hot[i, label] = 1
    #     return one_hot

    def encode_labels(self, labels):
        num_classes = 4  #num classes in calms21 dataset
        one_hot = np.zeros((len(labels), num_classes))
        for i, label in enumerate(labels):
            one_hot[i, label] = 1
        return one_hot

    def __getitem__(self, idx):
        if self.keypoints:
            sequence, labels, frame_ids, keypoints_sequence = self.data[idx]
        else:
            sequence, labels, frame_ids = self.data[idx]
        if self.max_seq_length:
            sequence = sequence[:self.max_seq_length]
            labels = labels[:self.max_seq_length]
            frame_ids = frame_ids[:self.max_seq_length]
            keypoints_sequence = keypoints_sequence[:self.max_seq_length]

        time_steps = np.arange(len(sequence)) / (len(sequence) - 1)  # Normalize to [0, 1]
        if self.keypoints:
            return {
                'time_steps': torch.FloatTensor(time_steps),
                'vals': torch.FloatTensor(sequence),
                'mask': torch.ones_like(torch.FloatTensor(sequence)),  # No missing data
                'labels': torch.FloatTensor(labels),
                'frame_ids':torch.LongTensor(frame_ids),
                'keypoints': torch.FloatTensor(keypoints_sequence)
            }
        else:
            return {
                'time_steps': torch.FloatTensor(time_steps),
                'vals': torch.FloatTensor(sequence),
                'mask': torch.ones_like(torch.FloatTensor(sequence)),  # No missing data
                'labels': torch.FloatTensor(labels),
                'frame_ids': torch.LongTensor(frame_ids)
            }


def variable_time_collate_fn_embeddings_keypoints(batch, args, device=torch.device("cpu"), data_type="train"):
    """
    Expects a batch of time series data in the form of (tt, vals, mask, labels) wher
        - tt is a 1-dimensional tensor containing T time values of observations.
        - vals is a (T, D) tensor containing observed values for D variables.
        - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
        - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
        combined_tt: The union of all time observations.
        combined_vals: (M, T, D) tensor containing the observed values.
        combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    D = batch[0]["vals"].shape[1]
    N = batch[0]["labels"].shape[1]  # number of labels

    K = batch[0]["keypoints"].shape[1]
    combined_tt, inverse_indices = torch.unique(torch.cat([ex["time_steps"] for ex in batch]), sorted=True, return_inverse=True)
    combined_tt = combined_tt.to(device)
    combined_frame_ids = torch.zeros([len(batch), len(combined_tt)], dtype=torch.long).to(device)
    K1, K2 = batch[0]["keypoints"].shape[1], batch[0]["keypoints"].shape[2]
    combined_keypoints = torch.zeros([len(batch), len(combined_tt), K1, K2]).to(device)

    offset = 0
    #combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    combined_vals = torch.zeros([len(batch), len(combined_tt), D + K1 * K2]).to(device)
    combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    combined_labels = torch.zeros([len(batch), len(combined_tt), N]).to(device)
    #combined_keypoints = torch.zeros([len(batch), len(combined_tt), K]).to(device)

    for b, (example) in enumerate(batch):
        tt = example['time_steps'].to(device)
        vals = example['vals'].to(device)
        mask = example['mask'].to(device)
        labels = example['labels'].to(device)
        frame_ids = example['frame_ids'].to(device)
        keypoints = example['keypoints'].to(device)

        flattened_keypoints = keypoints.view(keypoints.size(0), -1)
        combined_input = torch.cat([vals, flattened_keypoints], dim=-1)

        indices = inverse_indices[offset:offset + len(tt)]
        offset += len(tt)

        combined_vals[b, indices] = combined_input
        combined_mask[b, indices] = mask
        combined_labels[b, indices] = labels
        combined_frame_ids[b, indices] = frame_ids
        combined_keypoints[b, indices] = keypoints

    combined_tt = combined_tt.float()

    if torch.max(combined_tt) != 0.:
        combined_tt = combined_tt / torch.max(combined_tt)

    data_dict = {
        "time_steps": combined_tt,
        "data": combined_vals,
        "mask": combined_mask,
        "labels": combined_labels,
        "frame_ids": combined_frame_ids, #shape 50,50,1478 -> 1478 = 64 + 101*14
    }
    data_dict = utils.split_and_subsample_batch(data_dict, args, data_type=data_type)
    return data_dict


def variable_time_collate_fn_embeddings(batch, args, device=torch.device("cpu"), data_type="train"):
    """
    Expects a batch of time series data in the form of (tt, vals, mask, labels) wher
        - tt is a 1-dimensional tensor containing T time values of observations.
        - vals is a (T, D) tensor containing observed values for D variables.
        - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
        - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
        combined_tt: The union of all time observations.
        combined_vals: (M, T, D) tensor containing the observed values.
        combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    D = batch[0]["vals"].shape[1]
    N = batch[0]["labels"].shape[1]  # number of labels
    combined_tt, inverse_indices = torch.unique(torch.cat([ex["time_steps"] for ex in batch]), sorted=True, return_inverse=True)
    combined_tt = combined_tt.to(device)
    combined_frame_ids = torch.zeros([len(batch), len(combined_tt)], dtype=torch.long).to(device)


    offset = 0
    combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    combined_labels = torch.zeros([len(batch), len(combined_tt), N]).to(device)

    for b, (example) in enumerate(batch):
        tt = example['time_steps'].to(device)
        vals = example['vals'].to(device)
        mask = example['mask'].to(device)
        labels = example['labels'].to(device)
        frame_ids = example['frame_ids'].to(device)

        indices = inverse_indices[offset:offset + len(tt)]
        offset += len(tt)

        combined_vals[b, indices] = vals
        combined_mask[b, indices] = mask
        #

        combined_labels[b, indices] = labels
        combined_frame_ids[b, indices] = frame_ids

    combined_tt = combined_tt.float()

    if torch.max(combined_tt) != 0.:
        combined_tt = combined_tt / torch.max(combined_tt)

    data_dict = {
        "time_steps": combined_tt,
        "data": combined_vals,
        "mask": combined_mask,
        "labels": combined_labels,
        "frame_ids": combined_frame_ids
    }
    data_dict = utils.split_and_subsample_batch(data_dict, args, data_type=data_type)
    return data_dict
# if __name__ == '__main__':
#     torch.manual_seed(1991)
#
#     dataset = PersonActivity('data/PersonActivity', download=True)
#     dataloader = DataLoader(dataset, batch_size=30, shuffle=True, collate_fn=variable_time_collate_fn_activity)
#     dataloader.__iter__().next()
