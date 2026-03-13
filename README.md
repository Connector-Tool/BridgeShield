# BridgeShield
## 1. 安装环境
```
pip install -r requirements.txt
```



## 2. 数据处理

1. 将相匹配的源链和目标链上的原始的数据处理成交易对
```
python ctp.py
```
2. 将所有的交易对处理为pyg格式
```
python ctp_pyg.py
```



## 3. 模型运行

```
python BridgeShield.py
```
