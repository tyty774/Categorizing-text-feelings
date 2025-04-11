import torch
import torch.nn
import os
from getdata import Data,Mydataset,CustomDataLoader
from torch.nn import Module,Linear
from torch.optim import Adam
from torch.functional import F
from torch.nn import CrossEntropyLoss
import torch.nn
from train_test import Train_Test
from tqdm import tqdm
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained(r"C:\Users\aaa\.cache\huggingface\hub\datasets--NousResearch--hermes-function-calling-v1\snapshots")
model = DistilBertForSequenceClassification.from_pretrained(r"C:\Users\aaa\.cache\huggingface\hub\datasets--NousResearch--hermes-function-calling-v1\snapshots")
for param in model.parameters():
    param.requires_grad = False
def toke(data):
    # print(data)
    result_ids = [tokenizer(i,return_tensors='pt',max_length=512,padding='max_length',truncation=True)['input_ids'] for i in data]
    result_mask = [tokenizer(i,return_tensors='pt',max_length=512,padding='max_length',truncation=True)['attention_mask'] for i in data]
    return result_ids,result_mask



class train_model(Module):
    def __init__(self):
        super(train_model, self).__init__()
        self.model = model
        self.linear1 = Linear(2,100)
        self.linear2 = Linear(100,2)
    def forward(self,x,mask):
        x = x.squeeze()
        mask = mask.squeeze()
        out = self.model(input_ids=x,attention_mask = mask)
        out = out.logits
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        return out


if __name__=='__main__':

    loader = Data()
    # 获取训练数据
    train_texts, train_labels = loader.get_train_data()
    train_texts,train_mask = toke(train_texts)
    # 获取测试数据
    test_texts, test_labels = loader.get_test_data()
    test_texts,test_mask = toke(test_texts)

    train_test = Train_Test()

    train_dataset = Mydataset(train_texts,train_mask, train_labels)
    train_loader = CustomDataLoader(train_dataset).get_loader()
    ecpho = 10
    my_model = train_model()
    my_model.to(device=device)
    loss = CrossEntropyLoss()
    optimizer = Adam(my_model.parameters(),lr= 0.001)
    print('-------------------训练-----------------')
    train_test.train(train_loader,ecpho,my_model,loss,optimizer,device)
    print('-------------------预测-----------------')



    model.eval()
    test_dataset = Mydataset(test_texts,test_mask, test_labels)
    test_loader = CustomDataLoader(test_dataset).get_loader()
    train_test.test(test_loader,my_model,loss,device)

























# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# with torch.no_grad():
#     print(inputs)
#     logits = model(**inputs).logits
#
# predicted_class_id = logits.argmax().item()

