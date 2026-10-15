# sl-graph

StarLight 的静态计算图规划器，将算子拓扑转换为可执行的线性命令序列和静态内存布局。

## 职责边界

- 表达算子节点、张量形状和依赖关系。
- 执行图融合、临时资源标记与生命周期区间分析。
- 生成 `CommandBuffer` 和 `MemoryPlan`，为运行期复用 Scratchpad。

接口与实现仍在基线阶段；具体契约以 RFC-0001 为准。

## 开发

在 StarLight workspace 根目录运行 `cargo check -p sl-graph` 或 `cargo test -p sl-graph`。
