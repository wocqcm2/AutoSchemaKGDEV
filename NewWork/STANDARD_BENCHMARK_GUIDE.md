# 📊 使用AutoSchemaKG标准Benchmark测试NewWork KG

## 🎯 **概述**

这个指南帮你使用AutoSchemaKG的标准benchmark系统来科学评估你的NewWork概念图谱质量，获得**权威、可比较的评估指标**。

## 🏗️ **系统架构**

```
NewWork KG → Atlas格式转换 → 标准Retriever → 标准数据集测试 → 科学评估指标
```

## 🚀 **快速开始**

### 1. **运行标准benchmark**
```bash
cd NewWork/
python standard_benchmark_integration.py
```

### 2. **选择测试模式**
- **快速测试**: 10个样本，用于验证集成
- **标准测试**: 50个样本，平衡速度和准确性  
- **完整评估**: 200个样本，获得最可靠结果

## 📊 **评估指标说明**

### **核心指标**
| 指标 | 说明 | 范围 | 越高越好 |
|------|------|------|----------|
| **EM** | 精确匹配率 | 0-1 | ✅ |
| **F1** | F1分数 | 0-1 | ✅ |
| **Recall@2** | Top-2召回率 | 0-1 | ✅ |
| **Recall@5** | Top-5召回率 | 0-1 | ✅ |

### **性能基准**
- **优秀**: EM > 0.4, F1 > 0.5
- **良好**: EM > 0.25, F1 > 0.35  
- **基本**: EM > 0.15, F1 > 0.25

## 🔧 **测试的Retriever方法**

1. **SimpleGraphRetriever** - 基础图检索
2. **SimpleTextRetriever** - 文本检索
3. **TogRetriever** - Tree of Generation检索
4. **HippoRAGRetriever** - HippoRAG检索
5. **HippoRAG2Retriever** - HippoRAG2改进版

## 📁 **数据集信息**

### **Musique数据集**
- **类型**: 多跳问答数据集
- **问题数量**: 约25,000个
- **难度**: 中等到困难
- **特点**: 需要推理多个文档片段

### **数据格式示例**
```json
{
  "id": "2hop__13548_13529_para_0",
  "text": "问题相关的文本内容...",
  "metadata": {
    "question": "Messi的目标是什么时候？",
    "answer": "June 1982",
    "is_supporting": true
  }
}
```

## 📊 **结果解读**

### **输出文件位置**
```
./result/musique/
├── summary_YYYYMMDDHHMMSS_*.json     # 摘要指标
└── result_YYYYMMDDHHMMSS_*.json      # 详细结果
```

### **摘要文件格式**
```json
{
  "SimpleGraphRetriever_average_f1": 0.45,
  "SimpleGraphRetriever_average_em": 0.32,
  "SimpleGraphRetriever_average_recall@2": 0.58,
  "SimpleGraphRetriever_average_recall@5": 0.73,
  ...
}
```

## 🔍 **性能分析指南**

### **如果结果偏低 (<0.2)**
1. **检查数据质量**: 概念提取是否准确
2. **优化图结构**: 关系是否合理
3. **调整参数**: batch_size_concept, temperature等

### **如果某个Retriever表现特别差**
1. **数据兼容性**: 检查数据格式转换
2. **参数设置**: 调整特定retriever的参数
3. **模型匹配**: 确认embedding模型兼容

### **横向对比分析**
- **SimpleGraph vs SimpleText**: 图结构 vs 纯文本的效果
- **HippoRAG vs HippoRAG2**: 不同算法版本的性能
- **ToG**: 生成式检索的表现

## ⚡ **优化建议**

### **提升KG质量**
1. **概念提取优化**
   ```bash
   # 调整config.json中的参数
   "batch_size_concept": 4,        # 降低批次大小
   "temperature": 0.1,             # 提高确定性
   "min_concept_frequency": 2      # 过滤低频概念
   ```

2. **图结构优化**
   ```bash
   # 在direct_concept_config.py中调整
   "include_abstraction_levels": true,
   "include_hierarchical_relations": true,
   "filter_low_quality_concepts": true
   ```

### **Benchmark调优**
```python
# 在standard_benchmark_integration.py中调整
benchmark_config = BenchMarkConfig(
    number_of_samples=100,          # 增加样本数
    react_max_iterations=5,         # 增加ReAct迭代
)
```

## 🔄 **与现有测试的对比**

| 测试类型 | 数据集 | 评估标准 | 适用场景 |
|----------|--------|----------|----------|
| **advanced_rag_benchmark.py** | Dulce自定义 | 自定义查询 | 快速验证 |
| **standard_benchmark_integration.py** | Musique标准 | 标准指标 | 科学评估 ✅ |

## 🛠️ **故障排除**

### **常见问题**

1. **"数据文件未找到"**
   ```bash
   # 确保benchmark数据存在
   ls ../benchmark_data/musique*.json
   ```

2. **"Retriever创建失败"**
   ```bash
   # 检查依赖安装
   pip install sentence-transformers faiss-cpu
   ```

3. **"内存不足"**
   ```bash
   # 减少样本数量
   benchmark.run_standard_benchmark(num_samples=20)
   ```

## 🎯 **最佳实践**

1. **先运行快速测试**验证集成正确性
2. **逐步增加样本数**观察性能趋势
3. **对比多个数据集**确保方法泛化性
4. **保存测试配置**确保结果可复现
5. **定期重测**验证改进效果

## 📈 **进阶用法**

### **批量测试多个配置**
```python
configs = [
    {"temperature": 0.1, "samples": 50},
    {"temperature": 0.3, "samples": 50},
    {"temperature": 0.5, "samples": 50},
]

for config in configs:
    # 修改配置并运行测试
    run_with_config(config)
```

### **自定义评估指标**
```python
# 在QAJudger中添加自定义指标
def custom_metric(self, pred, ref):
    # 你的自定义评估逻辑
    return score
```

这套标准benchmark让你的NewWork KG获得**权威认证**，结果可以直接与其他AutoSchemaKG方法进行对比！🚀