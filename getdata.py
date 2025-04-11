import os
from torch.utils.data import Dataset,DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

class Data:
    def __init__(self, root_dir=r"D:\pythonProject\learn_torch\huggingface\transformers_test\data"):
        """
        初始化数据加载器
        :param root_dir: 根目录路径 (默认为"data")
        """
        self.root_dir = root_dir
        self.train_texts, self.train_labels = self._load_dataset("train")
        self.test_texts, self.test_labels = self._load_dataset("test")

    def _load_dataset(self, dataset_type):
        """
        加载指定类型的数据集 (train/test)
        :param dataset_type: 数据集类型 ("train" 或 "test")
        :return: (文本列表, 标签列表)
        """
        texts = []
        labels = []
        dataset_path = os.path.join(self.root_dir, dataset_type)

        # 遍历 neg 和 pos 目录
        for label_dir in ["neg", "pos"]:
            label = 1 if label_dir == "neg" else 0  # 设置标签
            dir_path = os.path.join(dataset_path, label_dir)

            # 遍历目录中的所有txt文件
            for filename in os.listdir(dir_path):
                if filename.endswith(".txt"):
                    file_path = os.path.join(dir_path, filename)

                    # 读取文件内容
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        texts.append(content)
                        labels.append(label)

        return texts, labels

    def get_train_data(self):
        """获取训练数据"""
        return self.train_texts, self.train_labels

    def get_test_data(self):
        """获取测试数据"""
        return self.test_texts, self.test_labels



class Mydataset(Dataset):
   def __init__(self,date,mask,label):
       super(Mydataset, self).__init__()
       self.data = date
       self.mask = mask
       self.label = label
   def __getitem__(self, item):
       return (self.data[item],self.mask[item],self.label[item])
   def __len__(self):
       return len(self.data)



class CustomDataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    @staticmethod
    def collate_fn(batch):
        """处理变长序列的collate函数，自动padding"""
        # 解压batch数据
        tensors = [item[0] for item in batch]
        masks = [item[1] for item in batch]
        labels = [item[2] for item in batch]

        # 自动padding（假设用0填充）
        padded_tensors = pad_sequence(tensors, batch_first=True, padding_value=0)
        padded_masks = pad_sequence(masks, batch_first=True, padding_value=0)

        # 转换labels为张量（假设labels是整数类别）
        label_tensor = torch.tensor(labels)

        return padded_tensors, padded_masks, label_tensor

    def get_loader(self):
        """创建DataLoader实例"""
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn,
            drop_last=True
        )
