// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! The compilation pipeline.
//!
//! The passes run in a fixed order:
//! `validate → specialize → capability → decompose → effects →
//! liveness/coloring → emit`.
//! See `docs/ARCH-0004` §10 and `docs/ARCH-0007` §3.3.

use std::collections::BTreeMap;

use sl_ir::{
    Command, CommandBuffer, CommandId, DType, DeviceCapabilities, Dim, Effect, IrError, MemoryPlan,
    Op, OpCode, Program, Shape, ShapeSpec, Slot, StateDecl, StateKind, Value, ValueId, ValueRole,
    IR_VERSION,
};

use crate::CompileError;

/// Compiles a symbolic program into a physical command buffer.
///
/// # Errors
///
/// Returns [`CompileError`] when the program is ill-formed, a symbolic extent
/// exceeds its bound, the backend lacks a required capability, an operator
/// cannot be decomposed, a state effect would be reordered, or the scratchpad
/// exceeds the backend limit.
pub fn compile(
    program: &Program,
    shapes: &ShapeSpec,
    capabilities: &DeviceCapabilities,
) -> Result<CommandBuffer, CompileError> {
    // 1. validate
    program.validate()?;

    let mut builder = Builder {
        program,
        shapes,
        capabilities,
        alignment: u64::from(capabilities.min_buffer_align.max(64)),
        values: Vec::new(),
        states: Vec::new(),
        commands: Vec::new(),
        origins: Vec::new(),
    };

    // 2. specialize
    builder.specialize()?;
    // 3. capability
    builder.check_capabilities()?;
    // 4. decompose
    builder.decompose()?;
    // 4b. backend shape constraints (applies to emitted commands)
    builder.check_constraints()?;
    // 5. effects
    builder.check_effects()?;
    // 6. liveness / coloring
    let plan = builder.plan_memory()?;
    // 7. emit (and self-check with `CommandBuffer::validate`)
    builder.emit(plan)
}

/// A `Temp` value with its command-index live range.
struct LiveTemp {
    value: ValueId,
    first: usize,
    last: usize,
    bytes: u64,
}

/// A scratchpad allocation that is still live.
struct ActiveSlot {
    offset: u64,
    bytes: u64,
    last: usize,
}

struct Builder<'a> {
    program: &'a Program,
    shapes: &'a ShapeSpec,
    capabilities: &'a DeviceCapabilities,
    alignment: u64,
    values: Vec<Value>,
    states: Vec<StateDecl>,
    commands: Vec<Command>,
    /// Source node index for every emitted command.
    origins: Vec<usize>,
}

impl Builder<'_> {
    fn specialize(&mut self) -> Result<(), CompileError> {
        let mut values = Vec::with_capacity(self.program.values.len());
        for value in &self.program.values {
            let shape = self.resolve_shape(&value.shape)?;
            values.push(Value::new(value.dtype, shape, value.role));
        }
        self.values = values;

        if self.capabilities.max_value_bytes != 0 {
            for (index, value) in self.values.iter().enumerate() {
                let bytes = value.byte_size()?;
                if bytes > self.capabilities.max_value_bytes {
                    return Err(CompileError::ValueTooLarge {
                        value: ValueId(index as u32),
                        bytes,
                        limit: self.capabilities.max_value_bytes,
                    });
                }
            }
        }

        let mut states = Vec::with_capacity(self.program.states.len());
        for state in &self.program.states {
            states.push(StateDecl {
                value: state.value,
                kind: self.resolve_kind(state.kind)?,
            });
        }
        self.states = states;

        for state in &self.states {
            let value =
                self.values
                    .get(state.value.0 as usize)
                    .ok_or(CompileError::Unimplemented(
                        "state value id is out of range",
                    ))?;
            let actual = value.shape.to_consts().ok_or(CompileError::Unimplemented(
                "symbolic state shape after specialization",
            ))?;
            let StateKind::KvCache {
                batch,
                layers,
                kv_heads,
                head_dim,
                max_seq,
                ..
            } = state.kind;
            let max_seq = max_seq.as_const().ok_or(CompileError::Unimplemented(
                "state max_seq was not specialized",
            ))?;
            let expected = vec![batch, layers, 2, max_seq, kv_heads, head_dim];
            if actual != expected {
                return Err(CompileError::InvalidStateShape {
                    value: state.value,
                    expected,
                    actual,
                });
            }
        }

        Ok(())
    }

    fn resolve_kind(&self, kind: StateKind) -> Result<StateKind, CompileError> {
        match kind {
            StateKind::KvCache {
                batch,
                layers,
                kv_heads,
                head_dim,
                max_seq,
                dtype,
            } => {
                let max_seq = self.resolve_dim(max_seq)?;
                Ok(StateKind::KvCache {
                    batch,
                    layers,
                    kv_heads,
                    head_dim,
                    max_seq: Dim::Const(max_seq),
                    dtype,
                })
            }
        }
    }

    fn resolve_shape(&self, shape: &Shape) -> Result<Shape, CompileError> {
        let mut dims = Vec::with_capacity(shape.rank());
        for dim in shape.as_slice() {
            dims.push(Dim::Const(self.resolve_dim(*dim)?));
        }
        Ok(Shape::new(dims))
    }

    fn resolve_dim(&self, dim: Dim) -> Result<u32, CompileError> {
        match dim {
            Dim::Const(value) => Ok(value),
            Dim::Sym(id) => {
                let entry = self
                    .shapes
                    .entries()
                    .iter()
                    .find(|entry| entry.id == id)
                    .ok_or(CompileError::Ir(IrError::UndeclaredSymbolicDim(id)))?;
                if entry.value > entry.max {
                    Err(CompileError::SymbolicDimExceedsMax {
                        dim: id,
                        max: entry.max,
                        requested: entry.value,
                    })
                } else {
                    Ok(entry.value)
                }
            }
        }
    }

    fn check_capabilities(&self) -> Result<(), CompileError> {
        self.capabilities.validate()?;

        for (node, instruction) in self.program.nodes.iter().enumerate() {
            let op = instruction.op.code();
            if op.is_core() {
                if !self.capabilities.core_ops.contains(op) {
                    return Err(CompileError::UnsupportedOp { node, op });
                }
            } else if !self.capabilities.composite_ops.contains(op) {
                match op {
                    // These composites have decomposition rules and are
                    // expanded in the next pass.
                    OpCode::SwiGlu | OpCode::RmsNormMatMul => {}
                    // `CausalAttention` deliberately has no decomposition rule
                    // in v1 (ARCH-0004 §5.3).
                    OpCode::CausalAttention => {
                        return Err(CompileError::UnsupportedOp { node, op });
                    }
                    other => {
                        return Err(CompileError::UnsupportedOp { node, op: other });
                    }
                }
            }
        }
        Ok(())
    }

    fn decompose(&mut self) -> Result<(), CompileError> {
        let program = self.program;
        for (index, node) in program.nodes.iter().enumerate() {
            self.emit_op(index, node.op, &node.inputs, &node.outputs, node.effect)?;
        }
        Ok(())
    }

    /// Emits a single program node, recursively expanding composites that the
    /// backend does not implement natively.
    ///
    /// `effect` is attached to the first command generated for the node so that
    /// the node's effect keeps its position relative to surrounding nodes.
    fn emit_op(
        &mut self,
        origin: usize,
        op: Op,
        inputs: &[ValueId],
        outputs: &[ValueId],
        effect: Effect,
    ) -> Result<(), CompileError> {
        let code = op.code();
        if code.is_core() || self.capabilities.composite_ops.contains(code) {
            if code.is_core() && !self.capabilities.core_ops.contains(code) {
                return Err(CompileError::UnsupportedOp {
                    node: origin,
                    op: code,
                });
            }
            self.push_command(op, inputs.to_vec(), outputs.to_vec(), effect, origin);
            return Ok(());
        }

        match op {
            Op::SwiGlu => self.expand_swi_glu(origin, inputs, outputs, effect),
            Op::RmsNormMatMul { eps } => {
                self.expand_rms_norm_matmul(origin, inputs, outputs, eps, effect)
            }
            _ => Err(CompileError::UnsupportedOp {
                node: origin,
                op: code,
            }),
        }
    }

    fn push_command(
        &mut self,
        op: Op,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
        effect: Effect,
        origin: usize,
    ) {
        let deps = if self.commands.is_empty() {
            Vec::new()
        } else {
            vec![CommandId((self.commands.len() - 1) as u32)]
        };
        self.commands.push(Command {
            op,
            inputs,
            outputs,
            deps,
            effect,
        });
        self.origins.push(origin);
    }

    fn push_temp(&mut self, shape: Shape) -> ValueId {
        let id = ValueId(self.values.len() as u32);
        self.values
            .push(Value::new(DType::F32, shape, ValueRole::Temp));
        id
    }

    fn concrete_dims(&self, value: ValueId) -> Result<Vec<u32>, CompileError> {
        let value = self
            .values
            .get(value.0 as usize)
            .ok_or(CompileError::Ir(IrError::UnknownValue(value)))?;
        value.shape.to_consts().ok_or(CompileError::Unimplemented(
            "a shape remained symbolic after specialization",
        ))
    }

    fn check_arity(
        &self,
        origin: usize,
        op: OpCode,
        inputs: &[ValueId],
        outputs: &[ValueId],
    ) -> Result<(), CompileError> {
        let signature = op.signature();
        if inputs.len() != signature.inputs.len() {
            return Err(CompileError::Ir(IrError::ArityMismatch {
                node: origin,
                expected: signature.inputs.len(),
                actual: inputs.len(),
            }));
        }
        if outputs.len() != signature.outputs.len() {
            return Err(CompileError::Ir(IrError::ArityMismatch {
                node: origin,
                expected: signature.outputs.len(),
                actual: outputs.len(),
            }));
        }
        Ok(())
    }

    fn check_port_rank(
        &self,
        origin: usize,
        port: usize,
        dims: &[u32],
        min: u8,
        max: u8,
    ) -> Result<(), CompileError> {
        let rank = dims.len();
        if rank < usize::from(min) || rank > usize::from(max) {
            return Err(CompileError::Ir(IrError::RankMismatch {
                node: origin,
                port,
                rank,
                min,
                max,
            }));
        }
        Ok(())
    }

    fn check_output_shape(
        &self,
        origin: usize,
        declared: &[u32],
        expected: &[u32],
    ) -> Result<(), CompileError> {
        if declared.len() != expected.len() {
            return Err(CompileError::Ir(IrError::RankMismatch {
                node: origin,
                port: 0,
                rank: declared.len(),
                min: expected.len() as u8,
                max: expected.len() as u8,
            }));
        }
        for (axis, (actual, expected)) in declared.iter().zip(expected).enumerate() {
            if actual != expected {
                let constraint = match axis {
                    0 => "decomposed output axis 0",
                    1 => "decomposed output axis 1",
                    2 => "decomposed output axis 2",
                    _ => "decomposed output axis 3",
                };
                return Err(CompileError::CapabilityMismatch {
                    constraint,
                    limit: *expected,
                    actual: *actual,
                });
            }
        }
        Ok(())
    }

    fn expand_swi_glu(
        &mut self,
        origin: usize,
        inputs: &[ValueId],
        outputs: &[ValueId],
        effect: Effect,
    ) -> Result<(), CompileError> {
        self.check_arity(origin, OpCode::SwiGlu, inputs, outputs)?;

        let x = self.concrete_dims(inputs[0])?;
        let gate = self.concrete_dims(inputs[1])?;
        let up = self.concrete_dims(inputs[2])?;
        let down = self.concrete_dims(inputs[3])?;

        self.check_port_rank(origin, 0, &x, 2, 4)?;
        self.check_port_rank(origin, 1, &gate, 2, 2)?;
        self.check_port_rank(origin, 2, &up, 2, 2)?;
        self.check_port_rank(origin, 3, &down, 2, 2)?;

        let (hidden, prefix) = x
            .split_last()
            .ok_or(CompileError::Ir(IrError::RankMismatch {
                node: origin,
                port: 0,
                rank: 0,
                min: 2,
                max: 4,
            }))?;
        let hidden = *hidden;
        let inter = gate[0];

        if gate[1] != hidden {
            return Err(CompileError::CapabilityMismatch {
                constraint: "SwiGlu w_gate hidden dimension",
                limit: hidden,
                actual: gate[1],
            });
        }
        if up[1] != hidden {
            return Err(CompileError::CapabilityMismatch {
                constraint: "SwiGlu w_up hidden dimension",
                limit: hidden,
                actual: up[1],
            });
        }
        if down[0] != hidden {
            return Err(CompileError::CapabilityMismatch {
                constraint: "SwiGlu w_down hidden dimension",
                limit: hidden,
                actual: down[0],
            });
        }
        if down[1] != inter {
            return Err(CompileError::CapabilityMismatch {
                constraint: "SwiGlu w_down intermediate dimension",
                limit: inter,
                actual: down[1],
            });
        }

        let mut intermediate_shape = prefix.to_vec();
        intermediate_shape.push(inter);
        let mut output_shape = prefix.to_vec();
        output_shape.push(hidden);
        let declared = self.concrete_dims(outputs[0])?;
        self.check_output_shape(origin, &declared, &output_shape)?;

        let gate_out = self.push_temp(Shape::consts(&intermediate_shape));
        let up_out = self.push_temp(Shape::consts(&intermediate_shape));
        let act_out = self.push_temp(Shape::consts(&intermediate_shape));
        let hidden_out = self.push_temp(Shape::consts(&intermediate_shape));

        // gate = MatMulNT(x, w_gate)
        self.emit_op(
            origin,
            Op::MatMulNT,
            &[inputs[0], inputs[1]],
            &[gate_out],
            effect,
        )?;
        // up = MatMulNT(x, w_up)
        self.emit_op(
            origin,
            Op::MatMulNT,
            &[inputs[0], inputs[2]],
            &[up_out],
            Effect::None,
        )?;
        // act = Silu(gate)
        self.emit_op(origin, Op::Silu, &[gate_out], &[act_out], Effect::None)?;
        // h = Mul(act, up)
        self.emit_op(
            origin,
            Op::Mul,
            &[act_out, up_out],
            &[hidden_out],
            Effect::None,
        )?;
        // out = MatMulNT(h, w_down)
        self.emit_op(
            origin,
            Op::MatMulNT,
            &[hidden_out, inputs[3]],
            &[outputs[0]],
            Effect::None,
        )?;
        Ok(())
    }

    fn expand_rms_norm_matmul(
        &mut self,
        origin: usize,
        inputs: &[ValueId],
        outputs: &[ValueId],
        eps: f32,
        effect: Effect,
    ) -> Result<(), CompileError> {
        self.check_arity(origin, OpCode::RmsNormMatMul, inputs, outputs)?;

        let x = self.concrete_dims(inputs[0])?;
        let norm_weight = self.concrete_dims(inputs[1])?;
        let weight = self.concrete_dims(inputs[2])?;

        self.check_port_rank(origin, 0, &x, 2, 4)?;
        self.check_port_rank(origin, 1, &norm_weight, 1, 1)?;
        self.check_port_rank(origin, 2, &weight, 2, 2)?;

        let (hidden, prefix) = x
            .split_last()
            .ok_or(CompileError::Ir(IrError::RankMismatch {
                node: origin,
                port: 0,
                rank: 0,
                min: 2,
                max: 4,
            }))?;
        let hidden = *hidden;
        let out = weight[0];

        if norm_weight[0] != hidden {
            return Err(CompileError::CapabilityMismatch {
                constraint: "RmsNormMatMul w_norm hidden dimension",
                limit: hidden,
                actual: norm_weight[0],
            });
        }
        if weight[1] != hidden {
            return Err(CompileError::CapabilityMismatch {
                constraint: "RmsNormMatMul w hidden dimension",
                limit: hidden,
                actual: weight[1],
            });
        }

        let mut output_shape = prefix.to_vec();
        output_shape.push(out);
        let declared = self.concrete_dims(outputs[0])?;
        self.check_output_shape(origin, &declared, &output_shape)?;

        let normalized = self.push_temp(Shape::consts(&x));
        // norm = RmsNorm(x, w_norm)
        self.emit_op(
            origin,
            Op::RmsNorm { eps },
            &[inputs[0], inputs[1]],
            &[normalized],
            effect,
        )?;
        // out = MatMulNT(norm, w)
        self.emit_op(
            origin,
            Op::MatMulNT,
            &[normalized, inputs[2]],
            &[outputs[0]],
            Effect::None,
        )?;
        Ok(())
    }

    fn check_effects(&self) -> Result<(), CompileError> {
        let mut last_write: BTreeMap<ValueId, usize> = BTreeMap::new();
        for (command, origin) in self.commands.iter().zip(&self.origins) {
            if let Effect::WriteState(state) = command.effect {
                if let Some(first) = last_write.get(&state) {
                    if *first > *origin {
                        return Err(CompileError::StateEffectConflict {
                            state,
                            first: *first,
                            second: *origin,
                        });
                    }
                }
                last_write.insert(state, *origin);
            }
        }
        Ok(())
    }

    fn check_constraints(&self) -> Result<(), CompileError> {
        for command in &self.commands {
            match command.op {
                Op::MatMulNT | Op::MatMulNN => {
                    let multiple = self.capabilities.constraints.matmul_k_multiple_of;
                    if multiple > 0 {
                        let dims = self.command_input_dims(command, 0)?;
                        let k = *dims.last().ok_or(CompileError::Unimplemented(
                            "matmul input has no reduction dimension",
                        ))?;
                        if k % multiple != 0 {
                            return Err(CompileError::CapabilityMismatch {
                                constraint: "matmul_k_multiple_of",
                                limit: multiple,
                                actual: k,
                            });
                        }
                    }
                }
                Op::Rope { .. } => {
                    let dims = self.command_input_dims(command, 0)?;
                    let head_dim = *dims.last().ok_or(CompileError::Unimplemented(
                        "rope input has no head dimension",
                    ))?;
                    self.check_head_dim(head_dim)?;
                }
                Op::SplitHeads { head_dim, .. } | Op::MergeHeads { head_dim, .. } => {
                    self.check_head_dim(head_dim)?;
                }
                Op::CausalAttention {
                    heads,
                    kv_heads,
                    head_dim,
                    ..
                } => {
                    self.check_head_dim(head_dim)?;
                    self.check_heads(heads)?;
                    self.check_heads(kv_heads)?;
                }
                Op::KvAppend { .. } => {
                    let dims = self.command_input_dims(command, 0)?;
                    let head_dim = *dims.last().ok_or(CompileError::Unimplemented(
                        "kv append input has no head dimension",
                    ))?;
                    self.check_head_dim(head_dim)?;
                    if let Some(kv_heads) = dims.get(1).copied() {
                        self.check_heads(kv_heads)?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn check_head_dim(&self, head_dim: u32) -> Result<(), CompileError> {
        let multiple = self.capabilities.constraints.head_dim_multiple_of;
        if multiple > 0 && head_dim % multiple != 0 {
            return Err(CompileError::CapabilityMismatch {
                constraint: "head_dim_multiple_of",
                limit: multiple,
                actual: head_dim,
            });
        }
        Ok(())
    }

    fn check_heads(&self, heads: u32) -> Result<(), CompileError> {
        let max_heads = self.capabilities.constraints.max_heads;
        if max_heads > 0 && heads > max_heads {
            return Err(CompileError::CapabilityMismatch {
                constraint: "max_heads",
                limit: max_heads,
                actual: heads,
            });
        }
        Ok(())
    }

    fn command_input_dims(&self, command: &Command, port: usize) -> Result<Vec<u32>, CompileError> {
        let id = *command.inputs.get(port).ok_or(CompileError::Unimplemented(
            "operator is missing an input port",
        ))?;
        let value = self
            .values
            .get(id.0 as usize)
            .ok_or(CompileError::Unimplemented("value id is out of range"))?;
        value.shape.to_consts().ok_or(CompileError::Unimplemented(
            "symbolic shape after specialization",
        ))
    }

    fn plan_memory(&self) -> Result<MemoryPlan, CompileError> {
        let mut ranges: Vec<Option<(usize, usize)>> = vec![None; self.values.len()];
        for (index, command) in self.commands.iter().enumerate() {
            for value in command.inputs.iter().chain(command.outputs.iter()) {
                let slot = value.0 as usize;
                if slot >= ranges.len() {
                    return Err(CompileError::Ir(IrError::UnknownValue(*value)));
                }
                match ranges[slot] {
                    Some((first, _)) => ranges[slot] = Some((first, index)),
                    None => ranges[slot] = Some((index, index)),
                }
            }
        }

        let mut temps = Vec::new();
        for (index, value) in self.values.iter().enumerate() {
            if value.role != ValueRole::Temp {
                continue;
            }
            let Some((first, last)) = ranges[index] else {
                continue;
            };
            let bytes = value.byte_size().map_err(CompileError::Ir)?;
            temps.push(LiveTemp {
                value: ValueId(index as u32),
                first,
                last,
                bytes,
            });
        }
        // Deterministic greedy interval allocation: earliest first use first,
        // ties broken by value id.
        temps.sort_by_key(|temp| (temp.first, temp.value.0));

        let mut active: Vec<ActiveSlot> = Vec::new();
        let mut slots = Vec::with_capacity(temps.len());
        let mut scratchpad_bytes = 0u64;

        for temp in temps {
            // Closed intervals: a slot may only be reused once its last use
            // strictly precedes this value's first use.
            active.retain(|slot| slot.last >= temp.first);

            let mut offset = 0u64;
            loop {
                offset = align_up(offset, self.alignment).ok_or(CompileError::PlanOverflow {
                    required: u64::MAX,
                    limit: self.capabilities.max_scratchpad_bytes,
                })?;
                let end = offset
                    .checked_add(temp.bytes)
                    .ok_or(CompileError::PlanOverflow {
                        required: u64::MAX,
                        limit: self.capabilities.max_scratchpad_bytes,
                    })?;
                let collision = active
                    .iter()
                    .filter(|slot| slot.bytes > 0 && temp.bytes > 0)
                    .filter(|slot| {
                        offset < slot.offset.saturating_add(slot.bytes) && slot.offset < end
                    })
                    .map(|slot| slot.offset.saturating_add(slot.bytes))
                    .max();
                match collision {
                    Some(collision_end) => offset = collision_end,
                    None => break,
                }
            }

            let end = offset
                .checked_add(temp.bytes)
                .ok_or(CompileError::PlanOverflow {
                    required: u64::MAX,
                    limit: self.capabilities.max_scratchpad_bytes,
                })?;
            scratchpad_bytes = scratchpad_bytes.max(end);
            slots.push(Slot {
                value: temp.value,
                offset,
                bytes: temp.bytes,
            });
            active.push(ActiveSlot {
                offset,
                bytes: temp.bytes,
                last: temp.last,
            });
        }

        if scratchpad_bytes > self.capabilities.max_scratchpad_bytes {
            return Err(CompileError::PlanOverflow {
                required: scratchpad_bytes,
                limit: self.capabilities.max_scratchpad_bytes,
            });
        }

        slots.sort_by_key(|slot| slot.value.0);
        Ok(MemoryPlan {
            scratchpad_bytes,
            slots,
        })
    }

    fn emit(self, plan: MemoryPlan) -> Result<CommandBuffer, CompileError> {
        let alignment = self.alignment;
        let program = self.program;
        let buffer = CommandBuffer {
            ir_version: IR_VERSION,
            values: self.values,
            commands: self.commands,
            plan,
            inputs: program.inputs.clone(),
            outputs: program.outputs.clone(),
            states: self.states,
        };
        buffer.validate(alignment).map_err(|error| match error {
            IrError::PlanOverflow { required, limit } => {
                CompileError::PlanOverflow { required, limit }
            }
            other => CompileError::Ir(other),
        })?;
        Ok(buffer)
    }
}

fn align_up(value: u64, alignment: u64) -> Option<u64> {
    if alignment == 0 {
        return None;
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - remainder)
    }
}
