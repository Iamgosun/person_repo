import os
import pathlib
from typing import Sequence

import pytorch_lightning as L
from dotenv import load_dotenv

from .common import default_transform

# zero-shot downstream datasets
from .flowers102 import Flowers102DataModule
from .food101 import Food101DataModule
from .cifar10 import CIFAR10DataModule
from .cifar100 import CIFAR100DataModule
from .ucf101 import UCF101DataModule
from .sun397 import SUN397DataModule

# active learning downstream datasets
from .homeoffice import (
    HomeOfficeArtDataModule,
    HomeOfficeClipartDataModule,
    HomeOfficeProductDataModule,
    HomeOfficeRealWorldDataModule,
)
from .homeoffice_da import (
    HomeOfficeDAArtDataModule,
    HomeOfficeDAClipartDataModule,
    HomeOfficeDAProductDataModule,
    HomeOfficeDARealWorldDataModule,
)

from .imagenet_wds import ImagenetWDSModule
from .imagenet_1k import (
    Imagenet1kDataModule,
    Imagenet50DataModule,
    Imagenet100DataModule,
)

from .imagenet_r import ImagenetRDataModule
from .imagenet_sketch import ImagenetSketchDataModule

from .imagenet_da import (
    ImagenetDARenditionsDataModule,
    ImagenetDASketchDataModule,
)

# pretraining
from .laion400m import Laion400mDataModule



# 数据模块工厂（DataModule Factory），用于在深度学习项目中统一管理和创建不同数据集的数据加载器
# 可以通过一个简单的字符串（如 'cifar10'）来实例化对应的数据集，而无需关心每个数据集具体的加载细节

# 通用下游任务数据集：如 cifar10, flowers102, food101。
# 领域自适应数据集：如 homeoffice-* 和 imagenet-da-* 系列。
# 预训练数据集：如 laion400m。
# ImageNet 变体：如 imagenet-100, imagenet-r 等。


# 在创建时 (create)：当你需要某个具体的数据集时，只需调用 create 方法并传入数据集名称（如 'cifar10'）。
#工厂方法会：
# 根据名称从 SUPPORTED_MODULES 中找到对应的数据模块类。
# 自动拼接出该数据集的完整路径。
# 将初始化时设置的通用参数和创建时指定的特定参数（如 subset_indices，用于只加载部分数据）一起传给数据模块的构造函数，
# 最终返回一个配置好的 LightningDataModule 实例。

# 支持小样本学习 (Few-Shot Learning)
# 代码中包含一个 use_few_shot 参数。当这个参数被设置为 True 时，工厂会向数据模块传递额外的参数，
# 如 shots_per_class（每个类别使用多少样本）和 few_shot_sample_seed（采样随机种子）。




SUPPORTED_MODULES = {
    'laion400m': Laion400mDataModule,

    # downstream datasets
    'flowers102': Flowers102DataModule,
    'food101': Food101DataModule,
    'cifar10': CIFAR10DataModule,
    'cifar100': CIFAR100DataModule,
    'sun397': SUN397DataModule,
    'ucf101': UCF101DataModule,

    # homeoffice datasets
    'homeoffice-art': HomeOfficeArtDataModule,
    'homeoffice-clipart': HomeOfficeClipartDataModule,
    'homeoffice-product': HomeOfficeProductDataModule,
    'homeoffice-realworld': HomeOfficeRealWorldDataModule,

    'homeoffice-da-art': HomeOfficeDAArtDataModule,
    'homeoffice-da-clipart': HomeOfficeDAClipartDataModule,
    'homeoffice-da-product': HomeOfficeDAProductDataModule,
    'homeoffice-da-realworld': HomeOfficeDARealWorldDataModule,

    # imagenet datasets
    'imagenet-val-wds': ImagenetWDSModule,
    'imagenet': Imagenet1kDataModule,
    'imagenet-100': Imagenet100DataModule,
    'imagenet-50': Imagenet50DataModule,

    'imagenet-r': ImagenetRDataModule,
    'imagenet-sketch': ImagenetSketchDataModule,

    'imagenet-da-r': ImagenetDARenditionsDataModule,
    'imagenet-da-sketch': ImagenetDASketchDataModule,
}

class DataModuleFactory:
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4, 
        text_prompt: str = "An image of a {class_name}",
        train_transform=default_transform(image_size=244),
        test_transform=default_transform(image_size=244),
        shuffle_train: bool = True,
        base_path: str = None,
        shots_per_class: int = 10,
        use_few_shot: bool = False,
        few_shot_sample_seed: int = 42,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.text_prompt = text_prompt
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.shuffle_train = shuffle_train
        
        self.shots_per_class = shots_per_class
        self.use_few_shot = use_few_shot
        self.few_shot_sample_seed = few_shot_sample_seed

        self.base_path = base_path
        if self.base_path is None:
            load_dotenv()
            self.base_path = os.getenv("DATA_BASE_DIR")
            

    def create(self, dataset_name: str, subset_indices: Sequence[int] = None) -> L.LightningDataModule:
        if dataset_name in SUPPORTED_MODULES:
            module = SUPPORTED_MODULES[dataset_name]
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        data_dir = pathlib.Path(self.base_path) / module.DATASET_SUBDIR
        
        if self.use_few_shot:
            return module(
                data_dir=data_dir,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                text_prompt=self.text_prompt,
                train_transform=self.train_transform,
                test_transform=self.test_transform,
                shuffle_train=self.shuffle_train,
                subset_indices=subset_indices,
                shots_per_class = self.shots_per_class,
                few_shot_sample_seed = self.few_shot_sample_seed,
                use_few_shot = self.use_few_shot
            )
        else:
            return module(
                data_dir=data_dir,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                text_prompt=self.text_prompt,
                train_transform=self.train_transform,
                test_transform=self.test_transform,
                shuffle_train=self.shuffle_train,
                subset_indices=subset_indices,
            )

