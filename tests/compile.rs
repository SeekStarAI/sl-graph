// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! End-to-end tests for the `sl-graph` compilation pipeline.

use sl_graph::{compile, CompileError};
use sl_ir::{
    BackendTag, CommandBuffer, DType, DTypeSet, DeviceCapabilities, Dim, Effect, IrError, Node, Op,
    OpCode, OpSet, Program, Shape, ShapeConstraints, ShapeSpec, ShapeSpecEntry, Slot, StateDecl,
    StateKind, SymDimDecl, SymId, Value, ValueId, ValueRole, IR_VERSION,
};

const SEQ: SymId = SymId(0);

fn caps(core: &[OpCode], composite: &[OpCode]) -> DeviceCapabilities {
    DeviceCapabilities {
        backend: BackendTag("test"),
        ir_version: IR_VERSION,
        core_ops: OpSet::new(core.to_vec()),
        composite_ops: OpSet::new(composite.to_vec()),
        dtypes: DTypeSet::ANY,
        min_buffer_align: 64,
        max_scratchpad_bytes: 1 << 30,
        max_value_bytes: 1 << 28,
        constraints: ShapeConstraints::default(),
    }
}

fn f32_value(dims: &[u32], role: ValueRole) -> Value {
    Value::new(DType::F32, Shape::consts(dims), role)
}

fn u32_value(dims: &[u32], role: ValueRole) -> Value {
    Value::new(DType::U32, Shape::consts(dims), role)
}

fn node(op: Op, inputs: Vec<ValueId>, outputs: Vec<ValueId>) -> Node {
    Node {
        op,
        inputs,
        outputs,
        effect: Effect::None,
    }
}

fn seq_spec(value: u32, max: u32) -> ShapeSpec {
    ShapeSpec::new(vec![ShapeSpecEntry {
        id: SEQ,
        max,
        value,
    }])
}

fn linear_program() -> Program {
    // RmsNorm -> MatMulNT -> Add
    Program {
        sym_dims: Vec::new(),
        values: vec![
            f32_value(&[2, 4], ValueRole::Input),  // 0: x
            f32_value(&[4], ValueRole::Param),     // 1: norm weight
            f32_value(&[3, 4], ValueRole::Param),  // 2: projection
            f32_value(&[2, 4], ValueRole::Temp),   // 3: normed
            f32_value(&[2, 3], ValueRole::Temp),   // 4: projected
            f32_value(&[2, 3], ValueRole::Input),  // 5: residual
            f32_value(&[2, 3], ValueRole::Output), // 6: result
        ],
        nodes: vec![
            node(
                Op::RmsNorm { eps: 1e-5 },
                vec![ValueId(0), ValueId(1)],
                vec![ValueId(3)],
            ),
            node(Op::MatMulNT, vec![ValueId(3), ValueId(2)], vec![ValueId(4)]),
            node(Op::Add, vec![ValueId(4), ValueId(5)], vec![ValueId(6)]),
        ],
        inputs: vec![ValueId(0), ValueId(5)],
        outputs: vec![ValueId(6)],
        states: Vec::new(),
    }
}

fn swi_glu_program() -> Program {
    Program {
        sym_dims: Vec::new(),
        values: vec![
            f32_value(&[2, 4], ValueRole::Input),  // 0: x
            f32_value(&[3, 4], ValueRole::Param),  // 1: w_gate
            f32_value(&[3, 4], ValueRole::Param),  // 2: w_up
            f32_value(&[4, 3], ValueRole::Param),  // 3: w_down
            f32_value(&[2, 4], ValueRole::Output), // 4: out
        ],
        nodes: vec![node(
            Op::SwiGlu,
            vec![ValueId(0), ValueId(1), ValueId(2), ValueId(3)],
            vec![ValueId(4)],
        )],
        inputs: vec![ValueId(0)],
        outputs: vec![ValueId(4)],
        states: Vec::new(),
    }
}

fn rms_norm_matmul_program() -> Program {
    Program {
        sym_dims: Vec::new(),
        values: vec![
            f32_value(&[2, 4], ValueRole::Input),  // 0: x
            f32_value(&[4], ValueRole::Param),     // 1: w_norm
            f32_value(&[3, 4], ValueRole::Param),  // 2: w
            f32_value(&[2, 3], ValueRole::Output), // 3: out
        ],
        nodes: vec![node(
            Op::RmsNormMatMul { eps: 1e-5 },
            vec![ValueId(0), ValueId(1), ValueId(2)],
            vec![ValueId(3)],
        )],
        inputs: vec![ValueId(0)],
        outputs: vec![ValueId(3)],
        states: Vec::new(),
    }
}

#[test]
fn compiles_linear_graph_and_passes_buffer_validation() {
    let program = linear_program();
    assert_eq!(program.validate(), Ok(()));

    let capabilities = caps(&[OpCode::RmsNorm, OpCode::MatMulNT, OpCode::Add], &[]);
    let buffer = compile(&program, &ShapeSpec::default(), &capabilities)
        .expect("the linear graph must compile");

    assert_eq!(buffer.commands.len(), 3);
    assert_eq!(buffer.plan.slots.len(), 2);
    assert_eq!(buffer.outputs, vec![ValueId(6)]);
    assert_eq!(buffer.inputs, vec![ValueId(0), ValueId(5)]);
    assert_eq!(buffer.validate(64), Ok(()));

    let ops: Vec<OpCode> = buffer.commands.iter().map(|c| c.op.code()).collect();
    assert_eq!(ops, vec![OpCode::RmsNorm, OpCode::MatMulNT, OpCode::Add]);
}

#[test]
fn compiles_the_decoder_fixture_graph() {
    // Gather -> RmsNorm -> MatMulNT -> LastToken, mirroring the runtime
    // fixture in ARCH-0007 §1.
    let seq_shape = Shape::new(vec![Dim::Const(1), Dim::Sym(SEQ)]);
    let seq_hidden = Shape::new(vec![Dim::Const(1), Dim::Sym(SEQ), Dim::Const(4)]);
    let seq_vocab = Shape::new(vec![Dim::Const(1), Dim::Sym(SEQ), Dim::Const(8)]);
    let program = Program {
        sym_dims: vec![SymDimDecl {
            id: SEQ,
            name: "seq",
            max: 8,
        }],
        values: vec![
            Value::new(DType::U32, seq_shape, ValueRole::Input),
            f32_value(&[8, 4], ValueRole::Param),
            Value::new(DType::F32, seq_hidden.clone(), ValueRole::Temp),
            f32_value(&[4], ValueRole::Param),
            Value::new(DType::F32, seq_hidden, ValueRole::Temp),
            f32_value(&[8, 4], ValueRole::Param),
            Value::new(DType::F32, seq_vocab, ValueRole::Temp),
            Value::new(DType::U32, Shape::default(), ValueRole::Scalar),
            f32_value(&[1, 8], ValueRole::Output),
        ],
        nodes: vec![
            node(Op::Gather, vec![ValueId(1), ValueId(0)], vec![ValueId(2)]),
            node(
                Op::RmsNorm { eps: 1e-5 },
                vec![ValueId(2), ValueId(3)],
                vec![ValueId(4)],
            ),
            node(Op::MatMulNT, vec![ValueId(4), ValueId(5)], vec![ValueId(6)]),
            node(
                Op::LastToken,
                vec![ValueId(6), ValueId(7)],
                vec![ValueId(8)],
            ),
        ],
        inputs: vec![ValueId(0)],
        outputs: vec![ValueId(8)],
        states: Vec::new(),
    };
    assert_eq!(program.validate(), Ok(()));

    let capabilities = caps(
        &[
            OpCode::Gather,
            OpCode::RmsNorm,
            OpCode::MatMulNT,
            OpCode::LastToken,
        ],
        &[],
    );
    let buffer = compile(&program, &seq_spec(2, 8), &capabilities)
        .expect("the decoder fixture must compile");

    assert_eq!(buffer.commands.len(), 4);
    assert_eq!(buffer.plan.slots.len(), 3);
    assert_eq!(
        buffer.value(ValueId(8)).unwrap().shape.to_consts(),
        Some(vec![1, 8])
    );
    assert!(!buffer.values.iter().any(|value| value.shape.has_symbolic()));
    assert_eq!(buffer.validate(64), Ok(()));
}

#[test]
fn specializes_symbolic_dimensions() {
    let program = Program {
        sym_dims: vec![SymDimDecl {
            id: SEQ,
            name: "seq",
            max: 8,
        }],
        values: vec![
            Value::new(
                DType::F32,
                Shape::new(vec![Dim::Const(2), Dim::Sym(SEQ)]),
                ValueRole::Input,
            ),
            Value::new(
                DType::F32,
                Shape::new(vec![Dim::Const(2), Dim::Sym(SEQ)]),
                ValueRole::Output,
            ),
        ],
        nodes: vec![node(
            Op::Scale { factor: 2.0 },
            vec![ValueId(0)],
            vec![ValueId(1)],
        )],
        inputs: vec![ValueId(0)],
        outputs: vec![ValueId(1)],
        states: Vec::new(),
    };
    assert_eq!(program.validate(), Ok(()));

    let capabilities = caps(&[OpCode::Scale], &[]);
    let buffer =
        compile(&program, &seq_spec(4, 8), &capabilities).expect("seq = 4 is within the bound");
    assert_eq!(
        buffer.value(ValueId(0)).unwrap().shape.to_consts(),
        Some(vec![2, 4])
    );
    assert_eq!(
        buffer.value(ValueId(1)).unwrap().shape.to_consts(),
        Some(vec![2, 4])
    );
    assert!(!buffer.values.iter().any(|v| v.shape.has_symbolic()));
    assert_eq!(buffer.validate(64), Ok(()));
}

#[test]
fn rejects_symbolic_dimensions_above_the_bound() {
    let program = Program {
        sym_dims: vec![SymDimDecl {
            id: SEQ,
            name: "seq",
            max: 8,
        }],
        values: vec![
            Value::new(
                DType::F32,
                Shape::new(vec![Dim::Const(2), Dim::Sym(SEQ)]),
                ValueRole::Input,
            ),
            Value::new(
                DType::F32,
                Shape::new(vec![Dim::Const(2), Dim::Sym(SEQ)]),
                ValueRole::Output,
            ),
        ],
        nodes: vec![node(
            Op::Scale { factor: 2.0 },
            vec![ValueId(0)],
            vec![ValueId(1)],
        )],
        inputs: vec![ValueId(0)],
        outputs: vec![ValueId(1)],
        states: Vec::new(),
    };

    let capabilities = caps(&[OpCode::Scale], &[]);
    let error = compile(&program, &seq_spec(9, 8), &capabilities).unwrap_err();
    assert_eq!(
        error,
        CompileError::SymbolicDimExceedsMax {
            dim: SEQ,
            max: 8,
            requested: 9,
        }
    );
}

#[test]
fn specializes_symbolic_kv_cache_max_seq() {
    let max_seq = Dim::Sym(SEQ);
    let program = Program {
        sym_dims: vec![SymDimDecl {
            id: SEQ,
            name: "seq",
            max: 8,
        }],
        values: vec![
            f32_value(&[1, 1, 2, 4], ValueRole::Input),
            f32_value(&[1, 1, 2, 4], ValueRole::Input),
            u32_value(&[], ValueRole::Scalar),
            Value::new(
                DType::F32,
                Shape::new(vec![
                    Dim::Const(1),
                    Dim::Const(1),
                    Dim::Const(2),
                    Dim::Sym(SEQ),
                    Dim::Const(1),
                    Dim::Const(4),
                ]),
                ValueRole::State,
            ),
        ],
        nodes: vec![Node {
            op: Op::KvAppend { layer: 0 },
            inputs: vec![ValueId(0), ValueId(1), ValueId(2)],
            outputs: Vec::new(),
            effect: Effect::WriteState(ValueId(3)),
        }],
        inputs: vec![ValueId(0), ValueId(1)],
        outputs: Vec::new(),
        states: vec![StateDecl {
            value: ValueId(3),
            kind: StateKind::KvCache {
                batch: 1,
                layers: 1,
                kv_heads: 1,
                head_dim: 4,
                max_seq,
                dtype: DType::F32,
            },
        }],
    };
    assert_eq!(program.validate(), Ok(()));

    let capabilities = caps(&[OpCode::KvAppend], &[]);
    let buffer = compile(&program, &seq_spec(4, 8), &capabilities)
        .expect("the KV append graph must compile");

    assert_eq!(buffer.commands.len(), 1);
    assert_eq!(buffer.commands[0].effect, Effect::WriteState(ValueId(3)));
    assert_eq!(
        buffer.value(ValueId(3)).unwrap().shape.to_consts(),
        Some(vec![1, 1, 2, 4, 1, 4])
    );
    match buffer.states[0].kind {
        StateKind::KvCache { max_seq, .. } => assert_eq!(max_seq, Dim::Const(4)),
    }
    assert_eq!(buffer.validate(64), Ok(()));
}

#[test]
fn rejects_core_op_missing_from_capabilities() {
    let program = Program {
        sym_dims: Vec::new(),
        values: vec![
            f32_value(&[4], ValueRole::Input),
            f32_value(&[4], ValueRole::Output),
        ],
        nodes: vec![node(
            Op::Add,
            vec![ValueId(0), ValueId(0)],
            vec![ValueId(1)],
        )],
        inputs: vec![ValueId(0)],
        outputs: vec![ValueId(1)],
        states: Vec::new(),
    };

    let capabilities = caps(&[OpCode::MatMulNT], &[]);
    let error = compile(&program, &ShapeSpec::default(), &capabilities).unwrap_err();
    assert_eq!(
        error,
        CompileError::UnsupportedOp {
            node: 0,
            op: OpCode::Add,
        }
    );
}

#[test]
fn rejects_causal_attention_without_native_support() {
    let program = Program {
        sym_dims: Vec::new(),
        values: vec![
            f32_value(&[1, 2, 2, 2], ValueRole::Input),
            f32_value(&[1, 2, 2, 2], ValueRole::Input),
            f32_value(&[1, 2, 2, 2], ValueRole::Input),
            u32_value(&[], ValueRole::Scalar),
            f32_value(&[1, 2, 2, 2], ValueRole::Output),
        ],
        nodes: vec![node(
            Op::CausalAttention {
                layer: 0,
                heads: 2,
                kv_heads: 2,
                head_dim: 2,
                scale: 0.5,
            },
            vec![ValueId(0), ValueId(1), ValueId(2), ValueId(3)],
            vec![ValueId(4)],
        )],
        inputs: vec![ValueId(0), ValueId(1), ValueId(2)],
        outputs: vec![ValueId(4)],
        states: Vec::new(),
    };
    assert_eq!(program.validate(), Ok(()));

    let capabilities = caps(&[], &[]);
    let error = compile(&program, &ShapeSpec::default(), &capabilities).unwrap_err();
    assert_eq!(
        error,
        CompileError::UnsupportedOp {
            node: 0,
            op: OpCode::CausalAttention,
        }
    );
}

#[test]
fn decomposes_swi_glu() {
    let program = swi_glu_program();
    assert_eq!(program.validate(), Ok(()));

    let capabilities = caps(&[OpCode::MatMulNT, OpCode::Silu, OpCode::Mul], &[]);
    let buffer =
        compile(&program, &ShapeSpec::default(), &capabilities).expect("SwiGlu must decompose");

    // 5 original values + 4 intermediates.
    assert_eq!(buffer.values.len(), 9);
    assert_eq!(buffer.commands.len(), 5);

    let ops: Vec<OpCode> = buffer.commands.iter().map(|c| c.op.code()).collect();
    assert_eq!(
        ops,
        vec![
            OpCode::MatMulNT,
            OpCode::MatMulNT,
            OpCode::Silu,
            OpCode::Mul,
            OpCode::MatMulNT,
        ]
    );

    // gate/up/act/h all have shape [2, 3]; the final command writes the
    // original output value with shape [2, 4].
    for id in 5..9 {
        let value = buffer.value(ValueId(id)).unwrap();
        assert_eq!(value.role, ValueRole::Temp);
        assert_eq!(value.shape, Shape::consts(&[2, 3]));
    }
    assert_eq!(
        buffer.value(ValueId(4)).unwrap().shape,
        Shape::consts(&[2, 4])
    );
    assert_eq!(buffer.commands[4].outputs, vec![ValueId(4)]);

    // Weights are passed through, not copied.
    assert_eq!(buffer.commands[0].inputs, vec![ValueId(0), ValueId(1)]);
    assert_eq!(buffer.commands[1].inputs, vec![ValueId(0), ValueId(2)]);
    assert_eq!(buffer.commands[4].inputs, vec![ValueId(8), ValueId(3)]);

    assert_eq!(buffer.plan.slots.len(), 4);
    assert_eq!(buffer.validate(64), Ok(()));
}

#[test]
fn keeps_swi_glu_when_the_backend_supports_it() {
    let program = swi_glu_program();
    let capabilities = caps(
        &[OpCode::MatMulNT, OpCode::Silu, OpCode::Mul],
        &[OpCode::SwiGlu],
    );
    let buffer = compile(&program, &ShapeSpec::default(), &capabilities)
        .expect("native SwiGlu must be retained");

    assert_eq!(buffer.commands.len(), 1);
    assert_eq!(buffer.values.len(), 5);
    assert_eq!(buffer.commands[0].op.code(), OpCode::SwiGlu);
    assert_eq!(buffer.validate(64), Ok(()));
}

#[test]
fn decomposes_rms_norm_matmul() {
    let program = rms_norm_matmul_program();
    assert_eq!(program.validate(), Ok(()));

    let capabilities = caps(&[OpCode::RmsNorm, OpCode::MatMulNT], &[]);
    let buffer = compile(&program, &ShapeSpec::default(), &capabilities)
        .expect("RmsNormMatMul must decompose");

    // 4 original values + 1 intermediate.
    assert_eq!(buffer.values.len(), 5);
    assert_eq!(buffer.commands.len(), 2);

    let ops: Vec<OpCode> = buffer.commands.iter().map(|c| c.op.code()).collect();
    assert_eq!(ops, vec![OpCode::RmsNorm, OpCode::MatMulNT]);

    let normalized = buffer.value(ValueId(4)).unwrap();
    assert_eq!(normalized.role, ValueRole::Temp);
    assert_eq!(normalized.shape, Shape::consts(&[2, 4]));
    assert_eq!(buffer.commands[1].inputs, vec![ValueId(4), ValueId(2)]);
    assert_eq!(
        buffer.value(ValueId(3)).unwrap().shape,
        Shape::consts(&[2, 3])
    );

    assert_eq!(buffer.plan.slots.len(), 1);
    assert_eq!(buffer.validate(64), Ok(()));
}

#[test]
fn non_overlapping_temps_share_one_slot() {
    let program = Program {
        sym_dims: Vec::new(),
        values: vec![
            f32_value(&[4], ValueRole::Input),
            f32_value(&[4], ValueRole::Temp),
            f32_value(&[4], ValueRole::Output),
            f32_value(&[4], ValueRole::Temp),
            f32_value(&[4], ValueRole::Output),
        ],
        nodes: vec![
            node(Op::Silu, vec![ValueId(0)], vec![ValueId(1)]),
            node(Op::Add, vec![ValueId(1), ValueId(0)], vec![ValueId(2)]),
            node(Op::Silu, vec![ValueId(0)], vec![ValueId(3)]),
            node(Op::Add, vec![ValueId(3), ValueId(0)], vec![ValueId(4)]),
        ],
        inputs: vec![ValueId(0)],
        outputs: vec![ValueId(2), ValueId(4)],
        states: Vec::new(),
    };

    let capabilities = caps(&[OpCode::Silu, OpCode::Add], &[]);
    let buffer = compile(&program, &ShapeSpec::default(), &capabilities).unwrap();

    let first = buffer
        .plan
        .slots
        .iter()
        .find(|slot| slot.value == ValueId(1))
        .unwrap();
    let second = buffer
        .plan
        .slots
        .iter()
        .find(|slot| slot.value == ValueId(3))
        .unwrap();
    assert_eq!(first.offset, second.offset);
    assert_eq!(first.bytes, 16);
    assert_eq!(second.bytes, 16);
    assert_eq!(buffer.plan.scratchpad_bytes, 16);
    assert_eq!(buffer.validate(64), Ok(()));
}

#[test]
fn chain_of_100_nodes_never_overlaps_live_temps() {
    let mut values = vec![f32_value(&[4], ValueRole::Input)];
    let mut nodes = Vec::new();

    // 99 chained temps followed by one Output, i.e. 100 commands.
    for index in 0..99u32 {
        let input = if index == 0 {
            ValueId(0)
        } else {
            ValueId(index)
        };
        values.push(f32_value(&[4], ValueRole::Temp));
        nodes.push(node(
            Op::Add,
            vec![input, ValueId(0)],
            vec![ValueId(index + 1)],
        ));
    }
    values.push(f32_value(&[4], ValueRole::Output));
    nodes.push(node(
        Op::Add,
        vec![ValueId(99), ValueId(0)],
        vec![ValueId(100)],
    ));

    let program = Program {
        sym_dims: Vec::new(),
        values,
        nodes,
        inputs: vec![ValueId(0)],
        outputs: vec![ValueId(100)],
        states: Vec::new(),
    };
    assert_eq!(program.nodes.len(), 100);
    assert_eq!(program.validate(), Ok(()));

    let capabilities = caps(&[OpCode::Add], &[]);
    let buffer = compile(&program, &ShapeSpec::default(), &capabilities)
        .expect("the chain graph must compile");
    assert_eq!(buffer.validate(64), Ok(()));

    let ranges = live_ranges(&buffer);
    assert_eq!(buffer.plan.slots.len(), 99);
    for slot in &buffer.plan.slots {
        assert!(ranges[slot.value.0 as usize].is_some());
    }

    let mut shared_offsets = 0usize;
    for (index, left) in buffer.plan.slots.iter().enumerate() {
        for right in &buffer.plan.slots[index + 1..] {
            if slot_offsets_overlap(left, right) {
                let (left_first, left_last) = ranges[left.value.0 as usize].unwrap();
                let (right_first, right_last) = ranges[right.value.0 as usize].unwrap();
                assert!(
                    left_last < right_first || right_last < left_first,
                    "values {} and {} share bytes while both are live",
                    left.value.0,
                    right.value.0
                );
                shared_offsets += 1;
            }
        }
    }
    assert!(
        shared_offsets > 0,
        "greedy coloring is expected to reuse scratchpad offsets"
    );

    let max_end = buffer
        .plan
        .slots
        .iter()
        .map(|slot| slot.offset + slot.bytes)
        .max()
        .unwrap_or(0);
    assert_eq!(buffer.plan.scratchpad_bytes, max_end);
}

#[test]
fn compilation_is_deterministic() {
    let program = swi_glu_program();
    let capabilities = caps(&[OpCode::MatMulNT, OpCode::Silu, OpCode::Mul], &[]);

    let first = compile(&program, &ShapeSpec::default(), &capabilities).unwrap();
    let second = compile(&program, &ShapeSpec::default(), &capabilities).unwrap();
    assert_eq!(format!("{first:?}"), format!("{second:?}"));
}

#[test]
fn scratchpad_covers_every_slot() {
    let program = swi_glu_program();
    let capabilities = caps(&[OpCode::MatMulNT, OpCode::Silu, OpCode::Mul], &[]);
    let buffer = compile(&program, &ShapeSpec::default(), &capabilities).unwrap();

    let mut max_end = 0;
    for slot in &buffer.plan.slots {
        assert_eq!(slot.offset % 64, 0);
        assert_eq!(
            slot.bytes,
            buffer.value(slot.value).unwrap().byte_size().unwrap()
        );
        let end = slot.offset + slot.bytes;
        assert!(end <= buffer.plan.scratchpad_bytes);
        max_end = max_end.max(end);
    }
    assert_eq!(buffer.plan.scratchpad_bytes, max_end);
    assert!(max_end > 0);
}

#[test]
fn honors_a_larger_backend_alignment() {
    let program = linear_program();
    let mut capabilities = caps(&[OpCode::RmsNorm, OpCode::MatMulNT, OpCode::Add], &[]);
    capabilities.min_buffer_align = 128;

    let buffer = compile(&program, &ShapeSpec::default(), &capabilities).unwrap();
    for slot in &buffer.plan.slots {
        assert_eq!(slot.offset % 128, 0);
    }
    assert_eq!(buffer.validate(128), Ok(()));
}

#[test]
fn rejects_scratchpad_above_backend_limit() {
    let program = linear_program();
    let mut capabilities = caps(&[OpCode::RmsNorm, OpCode::MatMulNT, OpCode::Add], &[]);
    capabilities.max_scratchpad_bytes = 64;

    let error = compile(&program, &ShapeSpec::default(), &capabilities).unwrap_err();
    match error {
        CompileError::PlanOverflow { required, limit } => {
            assert!(required > limit);
            assert_eq!(limit, 64);
        }
        other => panic!("expected PlanOverflow, got {other:?}"),
    }
}

#[test]
fn rejects_capabilities_with_sub_minimum_alignment() {
    let program = linear_program();
    let mut capabilities = caps(&[OpCode::RmsNorm, OpCode::MatMulNT, OpCode::Add], &[]);
    capabilities.min_buffer_align = 32;

    let error = compile(&program, &ShapeSpec::default(), &capabilities).unwrap_err();
    assert!(matches!(
        error,
        CompileError::Ir(IrError::InvalidCapabilities(_))
    ));
}

#[test]
fn rejects_values_above_the_backend_limit() {
    let program = linear_program();
    let mut capabilities = caps(&[OpCode::RmsNorm, OpCode::MatMulNT, OpCode::Add], &[]);
    capabilities.max_value_bytes = 8;

    let error = compile(&program, &ShapeSpec::default(), &capabilities).unwrap_err();
    assert!(matches!(error, CompileError::ValueTooLarge { .. }));
}

#[test]
fn enforces_matmul_reduction_constraints() {
    let program = Program {
        sym_dims: Vec::new(),
        values: vec![
            f32_value(&[2, 3], ValueRole::Input),
            f32_value(&[4, 3], ValueRole::Param),
            f32_value(&[2, 4], ValueRole::Output),
        ],
        nodes: vec![node(
            Op::MatMulNT,
            vec![ValueId(0), ValueId(1)],
            vec![ValueId(2)],
        )],
        inputs: vec![ValueId(0)],
        outputs: vec![ValueId(2)],
        states: Vec::new(),
    };
    assert_eq!(program.validate(), Ok(()));

    let mut capabilities = caps(&[OpCode::MatMulNT], &[]);
    capabilities.constraints.matmul_k_multiple_of = 4;
    let error = compile(&program, &ShapeSpec::default(), &capabilities).unwrap_err();
    assert_eq!(
        error,
        CompileError::CapabilityMismatch {
            constraint: "matmul_k_multiple_of",
            limit: 4,
            actual: 3,
        }
    );
}

#[test]
fn enforces_head_dimension_constraints() {
    let program = Program {
        sym_dims: Vec::new(),
        values: vec![
            f32_value(&[1, 2, 2, 2], ValueRole::Input),
            f32_value(&[1, 2, 2, 2], ValueRole::Input),
            f32_value(&[1, 2, 2, 2], ValueRole::Input),
            u32_value(&[], ValueRole::Scalar),
            f32_value(&[1, 2, 2, 2], ValueRole::Output),
        ],
        nodes: vec![node(
            Op::CausalAttention {
                layer: 0,
                heads: 2,
                kv_heads: 2,
                head_dim: 2,
                scale: 0.5,
            },
            vec![ValueId(0), ValueId(1), ValueId(2), ValueId(3)],
            vec![ValueId(4)],
        )],
        inputs: vec![ValueId(0), ValueId(1), ValueId(2)],
        outputs: vec![ValueId(4)],
        states: Vec::new(),
    };
    assert_eq!(program.validate(), Ok(()));

    let mut capabilities = caps(&[], &[OpCode::CausalAttention]);
    capabilities.constraints.head_dim_multiple_of = 4;
    let error = compile(&program, &ShapeSpec::default(), &capabilities).unwrap_err();
    assert_eq!(
        error,
        CompileError::CapabilityMismatch {
            constraint: "head_dim_multiple_of",
            limit: 4,
            actual: 2,
        }
    );
}

#[test]
fn rejects_state_shapes_that_disagree_with_the_kind() {
    let max_seq = Dim::Sym(SEQ);
    let program = Program {
        sym_dims: vec![SymDimDecl {
            id: SEQ,
            name: "seq",
            max: 8,
        }],
        values: vec![f32_value(&[1, 1, 2, 4, 1, 4], ValueRole::State)],
        nodes: Vec::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        states: vec![StateDecl {
            value: ValueId(0),
            kind: StateKind::KvCache {
                batch: 1,
                layers: 1,
                kv_heads: 1,
                head_dim: 8,
                max_seq,
                dtype: DType::F32,
            },
        }],
    };
    assert_eq!(program.validate(), Ok(()));

    let capabilities = caps(&[], &[]);
    let error = compile(&program, &seq_spec(4, 8), &capabilities).unwrap_err();
    assert_eq!(
        error,
        CompileError::InvalidStateShape {
            value: ValueId(0),
            expected: vec![1, 1, 2, 4, 1, 8],
            actual: vec![1, 1, 2, 4, 1, 4],
        }
    );
}

fn live_ranges(buffer: &CommandBuffer) -> Vec<Option<(usize, usize)>> {
    let mut ranges: Vec<Option<(usize, usize)>> = vec![None; buffer.values.len()];
    for (index, command) in buffer.commands.iter().enumerate() {
        for value in command.inputs.iter().chain(command.outputs.iter()) {
            let slot = value.0 as usize;
            match ranges[slot] {
                Some((first, _)) => ranges[slot] = Some((first, index)),
                None => ranges[slot] = Some((index, index)),
            }
        }
    }
    ranges
}

fn slot_offsets_overlap(left: &Slot, right: &Slot) -> bool {
    left.offset < right.offset + right.bytes && right.offset < left.offset + left.bytes
}
