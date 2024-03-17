# ebdl

E_BLIND/D_LC 系统的实现

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行测试

先将 data 解压到和本项目文件夹并列的文件夹：

```plain
- ebdl（本项目）
    - edbl
        - ebdl.py
    - test_1.py
    - ...
- data
    - 8.gif
    - ...
```

### 测试 1

```bash
python test_1.py
```

测试文件将保存在 `test1_result` 下（不包括统计图）。

### 测试 2

```bash
python test_2.py
```

测试文件将保存在 `test2_result` 下（不包括统计图）。

### 测试 3

```bash
python test_3.py
```

测试文件将保存在 `test3_result` 下（不包括统计图）。

### 另行测试

可以通过下列语句使用 ebdl：

```python
from ebdl import ebdl
```
