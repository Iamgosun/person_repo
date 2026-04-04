import torch
import pytorch_lightning as L
import datasets
from typing import Sequence
from .common import default_collate_fn, default_transform

# CIFAR10Dataset: 一个自定义的 PyTorch 数据集类，负责处理单个数据样本。
# CIFAR10DataModule: 一个 PyTorch Lightning 数据模块，负责管理整个数据集的生命周期（下载、划分、创建数据加载器）。


# CIFAR10Dataset 多模态输出: 一个字典。这个字典包含了：
# image: 经过 transform 处理后的图像张量。
# text: 根据 text_prompt 模板和类别名称生成的文本描述（例如 "An image of a cat"）。
# class_id: 图像的原始类别标签（整数）。
# image_id: 图像在数据集中的索引。



class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            data: datasets.Dataset,  # 接收一个由 datasets.load_dataset('cifar10') 加载的 Hugging Face datasets.Dataset 对象。
            text_prompt: str, 
            transform=None,
        ):
        self._data = data
        self._label_names = self._data.features['label'].names
        self._text_prompt = text_prompt
        self._transform = transform
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        text = self._text_prompt.format(
            class_name=self._label_names[self._data[idx]['label']]
        )

        image = self._data[idx]['img']
        if self._transform is not None:
            image = self._transform(image)
            
        return dict(image=image, text=text, class_id=self._data[idx]['label'], image_id=idx)



# CIFAR10DataModule 类详解
# 这个类继承自 L.LightningDataModule，是整个数据流程的总管。
# 1. 数据准备流程 (setup 方法)
# setup 方法是数据模块的核心，它在训练、验证或测试开始前被调用。
# 加载数据: 使用 datasets.load_dataset('cifar10', cache_dir=self.data_dir) 从 Hugging Face Hub 下载并加载 CIFAR10 数据集。
# 划分数据:
# 它将原始的 train 数据集再次划分，创建一个 80% 的训练集和一个 20% 的验证集。
# 原始的 test 数据集则作为最终的测试集。
# 创建数据集对象: 分别为训练、验证、测试集实例化 CIFAR10Dataset 对象，并传入相应的图像变换（train_transform 或 test_transform）。
# 支持子集采样: 如果传入了 subset_indices 参数，它会使用 torch.utils.data.Subset 来创建一个训练数据的子集。这在调试或进行少量样本实验时非常有用。
# 2. 创建数据加载器 (*_dataloader 方法)
# 这三个方法（train_dataloader, val_dataloader, test_dataloader）负责将 Dataset 对象包装成 DataLoader。
# DataLoader 配置: 它们都配置了 batch_size（批次大小）、num_workers（并行加载数据的进程数）等关键参数。
# persistent_workers=True: 这是一个性能优化选项，可以避免在每个 epoch 之间重复创建和销毁工作进程，从而加快数据加载速度。
# collate_fn=default_collate_fn: 指定了一个自定义的函数来将多个样本合并成一个批次。

class CIFAR10DataModule(L.LightningDataModule):
    DATASET_SUBDIR = 'cifar10'

    def __init__(
            self, 
            data_dir: str,
            batch_size: int = 32,
            num_workers: int = 4, 
            text_prompt: str = "An image of a {class_name}",
            train_transform=default_transform(image_size=244),
            test_transform=default_transform(image_size=244),
            shuffle_train: bool = True,
            # subset_indices 参数的作用是让你能够从整个数据集中，只选取指定索引位置的样本来构成一个新的、更小的数据集。
            subset_indices: Sequence[int] = None,
            # few shot parameters
            shots_per_class: int = 10,
            use_few_shot: bool = False,
            few_shot_sample_seed: int = 42,
        ):
        if use_few_shot:
            raise ValueError("Few shot not supported for this dataset")

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.text_prompt = text_prompt
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.shuffle_train = shuffle_train
        self.subset_indices = subset_indices

    def setup(self, stage: str = None):
        dataset = datasets.load_dataset('cifar10', cache_dir=self.data_dir)

        train_val_split = dataset['train'].train_test_split(test_size=0.2, seed=0)
        train_ds = train_val_split['train']
        val_ds = train_val_split['test']
        
        self.train_ds = CIFAR10Dataset(train_ds, text_prompt=self.text_prompt, transform=self.train_transform)
        if self.subset_indices is not None:
            self.train_ds = torch.utils.data.Subset(self.train_ds, self.subset_indices)

        self.val_ds = CIFAR10Dataset(val_ds, text_prompt=self.text_prompt, transform=self.test_transform)
        self.test_ds = CIFAR10Dataset(dataset['test'], text_prompt=self.text_prompt, transform=self.test_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle_train, 
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=default_collate_fn,
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=default_collate_fn,
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=default_collate_fn,
        )
    
    @property
    def class_prompts(self): #获取所有类别的文本提示列表
        return [self.text_prompt.format(class_name=name) for name in self.train_ds._label_names]