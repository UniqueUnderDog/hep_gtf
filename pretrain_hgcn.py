import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ZINC
from hgcn.hgcn_model import HGCNModel
from pathlib import Path  # 假设HGCNModel在此模块中定义



# 定义模型参数
C_VALUE = 1.0  # 曲率参数
DIM = 128  # 隐藏层维度
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.001
num_epochs = 100
# 创建模型实例
model = hgcn_model=HGCNModel(
                 c=1,
                 num_layers=12, 
                 dropout=0.1,
                 act="relu",
                 use_bias=True,
                 in_dim=1,
                 device=torch.device('cuda'),
                 cuda=0,
                 hidden_dim=16)  # 根据任务需求设置最终输出维度



# 定义损失函数和优化器
loss_fn= torch.nn.L1Loss()  # 分类任务常用交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 假设有如下数据集
train_dataset = ZINC("./data/ZINC",subset=True,split='train')
# 创建数据加载器
data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
device=torch.device("cuda")
model.to(device)
losses_per_epoch = []
# 开始训练循环
running_loss = 0.0
counter = 0

losses = []
for epoch in range(num_epochs):
    for i, batch in enumerate(data_loader):
        model.train()
        
        model.zero_grad()
        
        batch = batch.to(device)
        out= model(batch)
        loss = loss_fn(out.reshape(-1,), batch.y)
        loss.backward()
        
        loss_val = loss.item()
        running_loss += loss_val
        counter += 1
        
        if i == (len(data_loader))-1:
            mean_loss_val = running_loss / counter
            print(f'[{epoch}/{num_epochs-1}][{i}/{len(data_loader)}]|\t Loss:{mean_loss_val}')
            losses.append(mean_loss_val)
            counter = 0
            running_loss = 0
        optimizer.step()
        
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), losses, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Training Loss')
plt.title('Training Loss vs. Epochs')
plt.grid(True)
plt.show()


test_dataset = ZINC("./data/ZINC", subset=True, split='test')
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
model.eval()      
    
with torch.no_grad():  # 在评估模式下关闭梯度计算以提升效率
    test_running_loss = 0.0
    test_counter = 0
    for i, test_batch in enumerate(test_loader):
        test_batch = test_batch.to(device)
        test_out= model(test_batch)
        test_loss = loss_fn(test_out.reshape(-1,), test_batch.y)
        test_running_loss += test_loss.item()
        test_counter += 1

    test_mean_loss = test_running_loss / test_counter
    print(f'\nTest Loss after Training: {test_mean_loss}')


