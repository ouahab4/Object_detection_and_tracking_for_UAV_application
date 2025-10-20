import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class VisDroneSeqDetectionDataset(Dataset):
    def __init__(self, root, split="train", transforms=None):
        self.root = root
        self.sequences_dir = os.path.join(root, "sequences")
        self.annotations_dir = os.path.join(root, "annotations")
        self.sequence_names = sorted(os.listdir(self.sequences_dir))
        self.transforms = transforms

        self.samples = []
        print("Sequences found:", self.sequence_names)
        for seq in self.sequence_names:
            seq_dir = os.path.join(self.sequences_dir, seq)
            ann_file = os.path.join(self.annotations_dir, seq + ".txt")
            if not os.path.exists(ann_file):
                print(f"Missing annotation: {ann_file}")
                continue
            with open(ann_file, "r") as f:
                for line in f:
                    fields = line.strip().split(',')
                    if len(fields) < 8:
                        print(f"Malformed line in {ann_file}: {line.strip()}")
                        continue
                    frame_id = int(fields[0])
                    img_name = f"{frame_id:07d}.jpg"
                    img_path = os.path.join(seq_dir, img_name)
                    if not os.path.exists(img_path):
                        print(f"Missing image: {img_path}")
                    else:
                        self.samples.append((img_path, ann_file, frame_id))
        print(f"Total samples loaded: {len(self.samples)}")
                        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ann_file, frame_id = self.samples[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        boxes, labels = [], []
        with open(ann_file, "r") as f:
            for line in f:
                fields = line.strip().split(',')
                if int(fields[0]) != frame_id:
                    continue
                target_id = int(fields[1])
                x, y, w, h = map(float, fields[2:6])
                score = float(fields[6])
                category = int(fields[7])
                if category < 1 or category > 10 or target_id == -1:
                    continue
                boxes.append([x, y, x + w, y + h])
                labels.append(category - 1)  

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "orig_size": torch.tensor([orig_h, orig_w])
        }

        if self.transforms:
            img = self.transforms(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, target