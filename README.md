# sl-graph

StarLight 的静态计算图编译器：把 `sl_ir::Program`（符号图）编译为 `sl_ir::CommandBuffer`（物理命令流与静态内存布局）。

## 职责边界

- 校验符号程序与形状约束；
- 把符号维度具体化为编译期常量；
- 校验后端能力，并按需分解 `SwiGlu` / `RmsNormMatMul`；
- 校验 state 副作用与 `StateKind` 一致性；
- 计算 Temp 生命周期并做静态内存着色；
- 生成可被 `sl-core` 与 backend 执行的 `CommandBuffer`。

不负责：模型格式解析、模型家族 lowering、设备实现、请求调度。CPU 设备执行属于 `backend-*`，Engine 属于 `sl-core`。

## 公开 API

```rust
pub fn compile(
    program: &sl_ir::Program,
    shapes: &sl_ir::ShapeSpec,
    capabilities: &sl_ir::DeviceCapabilities,
) -> Result<sl_ir::CommandBuffer, CompileError>;
```

编译流水线顺序固定：

```text
validate → specialize → capability check → decompose → constraints →
effects → liveness/coloring → emit
```

关键规则：

- 输出中不得残留 `Dim::Sym`；`StateKind::KvCache::max_seq` 同样要被具体化；
- core op 必须出现在 `capabilities.core_ops` 中；
- composite 若未被后端原生支持，则按规则分解；`CausalAttention` 在 v1 没有分解规则，必须原生支持；
- `DeviceCapabilities::validate()` 必须通过（包括 `min_buffer_align >= 64`、非空 dtype 集合、`ir_version` 匹配）；
- `ShapeConstraints` 会检查 `head_dim_multiple_of`、`matmul_k_multiple_of`、`max_heads`；
- `max_value_bytes` 与 `max_scratchpad_bytes` 会分别产生 `ValueTooLarge` 与 `PlanOverflow`；
- 内存着色使用确定性的 first-fit：按 `(first_use, ValueId)` 排序，只复用生命周期不重叠的槽位，offset 按 `max(64, min_buffer_align)` 对齐；
- 同一输入编译两次，输出逐字节一致。

契约依据：[ARCH-0004](<../../docs/ARCH-0004: 计算图 IR 与设备命令 ABI 契约.md>) 与 [ARCH-0007](<../../docs/ARCH-0007: Graph Core Runtime 三线施工方案.md>)。

## 错误模型

`CompileError` 区分：

- `Ir`：符号程序本身不合法；
- `SymbolicDimExceedsMax`：符号维超出声明上界；
- `UnsupportedOp`：后端既未实现、也无法分解；
- `CapabilityMismatch`：后端约束不满足；
- `StateEffectConflict`：state 写入顺序冲突（防御性检查）；
- `PlanOverflow`：scratchpad 超出后端上限；
- `ValueTooLarge`：单个 value 超出后端上限；
- `InvalidStateShape`：state value 形状与 `StateKind` 不一致；
- `Unimplemented`：内部未实现路径。

## 开发

```sh
cargo test -p sl-graph
cargo clippy -p sl-graph --all-targets -- -D warnings
cargo fmt -p sl-graph --check
```

测试使用 `sl_ir` 手工构造 `Program` 与 `DeviceCapabilities`，不依赖任何 backend。当前 21 个集成测试，覆盖符号维、能力缺失、分解、着色、确定性、100 节点链与 decoder fixture 图。
