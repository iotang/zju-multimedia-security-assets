# eblk8

E_BLK_8/D_BLK_8 系统的实现

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行测试

先将 data 解压到和本项目文件夹并列的文件夹：

```plain
- eblk8（本项目）
    - eblk8
        - eblk8.py
    - test1_mp.py
    - ...
- data
    - 8.gif
    - ...
```

### 测试 1

**这些测试用时都会很长。**

如果测试用时太长，请删除 data 下的部分图片。

```bash
python test1_mp.py
```

测试文件将保存在 `test1_result` 下（不包括统计图）。

### 测试 2

```bash
python test2_mp.py
```

测试文件将保存在 `test2_result` 下（不包括统计图）。

### 测试 3

```bash
python test3_mp.py
```

测试文件将保存在 `test3_result` 下（不包括统计图）。

### 测试 4

```bash
python test4_mp.py
```

测试文件将保存在 `test4_result` 下（不包括统计图）。

### 测试 5

```bash
python test5_mp.py
```

测试文件将保存在 `test5_result` 下（不包括统计图）。

### 另行测试

可以通过下列语句使用 esimple8：

```python
from esimple8 import esimple8
```
