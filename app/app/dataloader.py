import torch
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

class TripletImageFolder(ImageFolder):
    """From the torchvision.datasets.ImageFolder it generates triplet samples, used in training. For testing we use normal image folder.
    Note: a triplet is composed by a pair of matching images and one of different class.
    """
    def __init__(self, *arg, **kw):
        super(TripletImageFolder, self).__init__(*arg, **kw)

        self.n_triplets = len(self.samples)
        self.train_triplets = self.generate_triplets()

    def generate_triplets(self):
        labels = torch.Tensor(self.targets)
        triplets = []
        for x in np.arange(self.n_triplets):
            idx = np.random.randint(0, labels.size(0))
            idx_matches = np.where(labels.numpy() == labels[idx].numpy())[0]
            idx_no_matches = np.where(labels.numpy() != labels[idx].numpy())[0]

            if len(idx_matches) < 2:
                continue

            idx_a, idx_p = np.random.choice(idx_matches, 2, replace=False)
            idx_n = np.random.choice(idx_no_matches, 1)[0]

            triplets.append([idx_a, idx_p, idx_n])

        return np.array(triplets)

    def set_triplets(self, triplets):
        self.train_triplets = triplets

    def __getitem__(self, index):
        t = self.train_triplets[index]

        path_a, _ = self.samples[t[0]]
        path_p, _ = self.samples[t[1]]
        path_n, _ = self.samples[t[2]]

        img_a = self.loader(path_a)
        img_p = self.loader(path_p)
        img_n = self.loader(path_n)

        if self.transform is not None:
            img_a = self.transform(img_a)
            img_p = self.transform(img_p)
            img_n = self.transform(img_n)

        return img_a, img_p, img_n
    

class TripletDataLoader:
    def __init__(self, dataset_path, batch_size=32, num_workers=1, image_size=224, is_shuffle=True):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.is_shuffle = is_shuffle
        self.dataset = TripletImageFolder(root=dataset_path, transform=self.transform)
        
        # TRANSFORM PARAMETERS
        self.brightness=0.2
        self.contrast=0.2
        self.saturation=0.2

    def _transform(self):
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def get_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, 
            shuffle=self.is_shuffle, num_workers=self.num_workers
        )
        return dataloader
    
# Example usage
if __name__ == "__main__":
    dataset_path = "path/to/your/dataset"
    batch_size = 32
    num_workers = 4
    image_size = 224

    triplet_dataloader = TripletDataLoader(dataset_path, batch_size, num_workers, image_size)
    dataloader = triplet_dataloader.get_dataloader()

    for batch in dataloader:
        img_a, img_p, img_n = batch
        print(f"Anchor: {img_a.shape}, Positive: {img_p.shape}, Negative: {img_n.shape}")
        break