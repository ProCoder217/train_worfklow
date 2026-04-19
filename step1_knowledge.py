import os, gc
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

SAVE_DIR = "./output"
os.makedirs(SAVE_DIR, exist_ok=True)
SOFT_PATH = f"{SAVE_DIR}/soft_labels.pt"

print("Downloading TrashNet dataset...")
dataset = load_dataset("garythung/trashnet")
raw_t = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

class WasteDS(Dataset):
    def __init__(self, s, t): self.s, self.t = s, t
    def __len__(self): return len(self.s)
    def __getitem__(self, i): return self.t(self.s[i]['image'].convert('RGB')), self.s[i]['label']

raw_loader = DataLoader(WasteDS(dataset['train'], raw_t), batch_size=32, num_workers=4)

print("Cloning nanoVLM and extracting features...")
os.system("git clone https://github.com/lusxvr/nanoVLM.git nanoVLM")
os.system(f"{__import__('sys').executable} -m pip install -q ./nanoVLM")

import sys; sys.path.insert(0, "nanoVLM")
from models.vision_language_model import VisionLanguageModel

teacher = VisionLanguageModel.from_pretrained("lusxvr/nanoVLM-222M").eval()
soft_label_map = {}

with torch.no_grad():
    for i, (imgs, lbls) in enumerate(raw_loader):
        pooled = teacher.vision_encoder(imgs).mean(dim=1)
        start = i * 32
        for j in range(len(lbls)):
            soft_label_map[start + j] = {'feature': pooled[j], 'label': lbls[j].item()}
        print(f"  Progress: {min(start+32, len(raw_loader.dataset))}/{len(raw_loader.dataset)}...", end='\r')

torch.save(soft_label_map, SOFT_PATH)
print(f"\n✅ Knowledge saved ({os.path.getsize(SOFT_PATH)/1024/1024:.1f} MB)")

del dataset, teacher, raw_loader; gc.collect()
