# LAB5 指南

By Benwei Wu, Jin Zhang, Yuanhao Pu

### 实验目标

考察完整进行实验的能力, 涵盖任务规划至执行各阶段, 不仅限于模型构建。


## 1. 实验原理

可参考本课程涉及的所有分类模型以及原理

## 2. Data

[README-dataset.md](README-dataset.md)

## 3. 任务、提示以及要求

### 3.1 任务

- **数据预处理**。需要注意, 数据可能包含冗余特征、outlier数据以及Null数据, 你可能需要进行数据补缺、特征提取、编码以及必要的其他数据预处理工作。
- **数据划分**。你需要将所提供的`train.json`数据集按照所学的方法拆分成训练集以及测试集。
- **模型训练**。你需要分别使用本课程所学习的线性回归模型、决策树模型、神经网络模型、支持向量机以及XGBoost等分类模型来完成标签预测任务。
- **模型验证**。你需要将`test.json`的数据输入到一个你认为性能最佳的模型中, 预测缺失的`label`字段, 将其保存, 并将它包含在你所提交的压缩包中。
- **实验分析**。你需要仔细撰写实验报告以及相关分析。

### 3.2 评价指标

- 你可以在实验中使用下列指标来验证你的算法效果以及不同参数对于结果的影响

  - $E(f;D)=\frac{1}{m}\sum_{i=1}^m\mathbb{I}(f(x_i)\neq y_i)$, 分类错误率, 越小越好
  - $Acc(f;D)=1-E(f;D)$, 分类精度, 越大越好
  - 运行时间
- 你也可以选取你认为合理的评价指标, 评价指标的选取不会影响最终的实验分数。
- 但需要注意, 在**模型验证**环节, 我们将使用$Acc(f;D)$指标对你的生成结果进行判分。当然, 这个分数只是最终实验分数的组成, 并不唯一, 因此我们鼓励你使用更多指标来综合评价模型性能。

### 3.3 提示

- 在进行实验时，对在前几次实验中已经实现的算法模型，应考虑直接应用; 对于尚未实现的算法，可以自主编写，也可以使用现成的算法库。实验的核心不仅仅是算法的复现，而是要关注整个实验流程的完整性和严谨性，这包括特征预处理、模型评估、参数调整、模型选择和假设检验等关键环节。实验报告的评价将主要基于这些内容，因此，请注意实验报告的叙述结构和可读性。鼓励同学们在实验报告中将自己认为在完成实验时出彩的部分进行**高亮标注**，或单独列出一个章节。具有建设性或启发性的尝试将有机会获得更高的分数。

- 本课程提供了多种基础模型，但也鼓励大家去探索和尝试课程以外的新型模型。请注意，只要能完成基本的实验任务，就能获得相应的分数，是否进行课程外方法的尝试拓展不会对总评成绩造成太大的影响。

- 在撰写实验报告时，应避免单纯堆砌数据和代码。更重要的是对实验结果进行深入的分析和详细的解释，以展示实验的意义和价值。

### 3.4 要求

- 你可以和其他同学讨论, 但是你不可以剽窃代码, 我们会用自动程序来判断代码之间的相似性, 一旦被发现, 你们两个本次实验都会得零分。
- 不要修改rawtest.json文件, 请务必确保将要提交的`test.json`文件数据格式是否符合规范, 运行
    ```
    python check.py
    ```
    检查文件是否规范。

## 4. 提交

* 报告推荐格式

  1. 实验目的（可选）
  2. 实验原理（若不重要可以简要说明）
  3. 实验步骤（从读取数据、模型训练、使用xx的参数, xx的模型, 得到了多少组的结果, 就是你在每块代码做了什么事情）
  4. 实验结果（对输出进行总结、比较、可视化）
  5. 实验分析（分析结果出现的原因、分析原因）
  6. 实验过程中的亮点（如有）

* 提交 .zip 文件, 包含以下内容（请直接对这三个文件打包）

  --exp.ipynb

  --test.json

  --report.pdf

* 请将您的文件命名为 LAB5_PBXXXXXXXX_中文名.zip，对于命名错误的文件，我们可能不予计分。

* 请在截止日期前将您的 .zip 文件发送至 ml2023fall@163.com。

* 截止日期：2023年2月4日 23:59:59