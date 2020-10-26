# 多层感知机

张清阳

## 定义网络

+ 继承torch.nn.Module搭建感知机
输入size为10，隐藏层size为20，输出size为10。
输入层和隐藏层之间全连接，经过激活函数得到隐藏层的输出，隐藏层输出和输出层全连接。
同时定义了前向传递函数。

+ 输入为x，随机生成；感知机输出为y；实际值为target，随机生成。

## 损失函数

调用torch.nn.MSELoss协方差损失函数

## 反向传播

调用backward，获得反向传播的误差

## 更新权重

调用torch.optim.SGD随机梯度下降，权重 = 权重 - 学习率 * 梯度

