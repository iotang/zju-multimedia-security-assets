# esimple8

E_SIMPLE_8/D_SIMPLE_8 系统的实现

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行测试

先将 data 解压到和本项目文件夹并列的文件夹：

```plain
- esimple8（本项目）
    - esimple8
        - esimple8.py
    - test1.py
    - ...
- data
    - 8.gif
    - ...
```

### 测试 1

```bash
python test1.py
```

测试文件将保存在 `test1_result` 下（不包括统计图）。

你也可以运行多线程版本 `test1_mp.py`，不过它的表现在大多数机器上都会更差。

### 测试 2

```bash
python test2.py
```

测试文件将保存在 `test2_result` 下（不包括统计图）。

如果测试用时太长，请删除 data 下的部分图片。

### 测试 3

**这个测试用时会很长。**

测试时会读取 `house.png`。

```bash
python test3.py
```

测试文件将保存在 `test3_result` 下（不包括统计图）。

如果测试用时太长，请删除 data 下的部分图片。

### 测试 4

```bash
python test4.py
```

测试时会读取 `rec2.bmp`。

测试文件将保存在 `test4_result` 下（不包括统计图）。

如果测试用时太长，请删除 data 下的部分图片。

### 另行测试

可以通过下列语句使用 esimple8：

```python
from esimple8 import esimple8
```
