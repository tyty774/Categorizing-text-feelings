from tqdm import tqdm
import time
import torch
class Train_Test():
    def train(self,train_loader,ecpho,my_model,loss,optimizer,device):
        for i in tqdm(range(ecpho)):
            acc_list = []
            loss_list = []
            for ind, (data, mask, label) in enumerate(train_loader):
                data = data.to(device=device)
                mask = mask.to(device=device)
                label = label.to(device=device)
                pre_label = my_model(data, mask)
                pre_label = pre_label.squeeze()
                label = label.reshape(-1).long()
                loss_val = loss(pre_label, label)
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                predicted_labels = torch.argmax(pre_label, dim=1)
                correct = (predicted_labels == label).sum().item()
                accuracy = correct / label.size(0)
                acc_list.append(accuracy)
                loss_list.append(loss_val)
            print(
                f"Train_Accuracy: {sum(acc_list) / len(acc_list) * 100:.2f}%,Train_loss:{sum(loss_list) / len(loss_list)}")
            time.sleep(1)
    def test(self,test_loader,my_model,loss,device):
        test_acc_list = []
        test_loss_list = []
        for ind, (data, mask, label) in tqdm(enumerate(test_loader)):
            data = data.to(device=device)
            mask = mask.to(device=device)
            label = label.to(device=device)
            pre_label = my_model(data, mask)
            pre_label = pre_label.squeeze()
            label = label.reshape(-1).long()
            loss_val = loss(pre_label, label)
            predicted_labels = torch.argmax(pre_label, dim=1)
            correct = (predicted_labels == label).sum().item()
            accuracy = correct / label.size(0)
            test_acc_list.append(accuracy)
            test_loss_list.append(loss_val)
        print(
            f"Test_Accuracy: {sum(test_acc_list) / len(test_acc_list) * 100:.2f}%,Test_loss:{sum(test_loss_list) / len(test_loss_list)}")
        time.sleep(1)