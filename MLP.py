#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn
import torch.nn.functional
import torch.optim


# In[2]:


#定义网络
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        
        #input--hidden
        self.fc1=torch.nn.Linear(10, 20)
        
        self.sigmoid_1=torch.nn.Sigmoid()
        #self.relu=torch.nn.ReLU()
        
        #hidden--output
        self.fc2=torch.nn.Linear(20, 10)
        
        
    def forward(self,x):
        #
        x=self.fc1(x)
        #print(x.size())
        
        x=self.sigmoid_1(x)
        #x=self.relu(x)
        #print(x.size())
        
        x=self.fc2(x)
        #print(x.size())
        
        return x


# In[3]:


mlp_1=MLP()
#print(mlp_1)


# In[4]:


#定义权重更新
optimizer=torch.optim.SGD(mlp_1.parameters(),lr=0.01)


# In[5]:


x=torch.randn(1,10)
target=torch.randn(1,10)
print(x)
print(target)


# In[6]:


loss=torch.tensor(1)
print(loss)
loss_fuction=torch.nn.MSELoss()

#for i in range(1000):
while loss>0.000001:
    y=mlp_1(x)
    
    #损失函数
    loss=loss_fuction(y,target)

    optimizer.zero_grad()

    #反向传播
    loss.backward()
    #更新梯度
    optimizer.step()


# In[8]:


print(type(loss))
print(loss.data)
print(y)

