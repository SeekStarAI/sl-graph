// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Errors produced by the graph compiler.

use core::fmt;

use sl_ir::{IrError, OpCode, SymId, ValueId};

/// Failure while compiling a [`sl_ir::Program`] into a
/// [`sl_ir::CommandBuffer`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompileError {
    /// The symbolic program itself is ill-formed.
    Ir(IrError),
    /// A symbolic dimension requests an extent above its declared bound.
    SymbolicDimExceedsMax {
        /// The offending symbolic dimension.
        dim: SymId,
        /// Upper bound from the shape specification.
        max: u32,
        /// Extent requested by the shape specification.
        requested: u32,
    },
    /// The backend neither implements the operator nor can the compiler
    /// decompose it.
    UnsupportedOp {
        /// Index of the source node.
        node: usize,
        /// The unsupported operator.
        op: OpCode,
    },
    /// A backend-specific limit was exceeded.
    CapabilityMismatch {
        /// Name of the violated constraint.
        constraint: &'static str,
        /// Value accepted by the backend.
        limit: u32,
        /// Value required by the program.
        actual: u32,
    },
    /// Two writes to the same state would be reordered.
    StateEffectConflict {
        /// The state value written by both nodes.
        state: ValueId,
        /// Source node of the earlier write.
        first: usize,
        /// Source node of the later write, which would execute first.
        second: usize,
    },
    /// The static scratchpad does not fit in the backend limit.
    PlanOverflow {
        /// Bytes required by the memory plan.
        required: u64,
        /// Bytes accepted by the backend.
        limit: u64,
    },
    /// A single value exceeds the backend's per-value byte limit.
    ValueTooLarge {
        /// The offending value.
        value: ValueId,
        /// Bytes required by the value.
        bytes: u64,
        /// Bytes accepted by the backend.
        limit: u64,
    },
    /// A state value's shape does not match its `StateKind` declaration.
    InvalidStateShape {
        /// The state value.
        value: ValueId,
        /// Shape implied by `StateKind`.
        expected: Vec<u32>,
        /// Shape declared by the program.
        actual: Vec<u32>,
    },
    /// A compiler path that is deliberately not implemented.
    Unimplemented(&'static str),
}

impl fmt::Display for CompileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ir(error) => write!(formatter, "invalid IR: {error}"),
            Self::SymbolicDimExceedsMax {
                dim,
                max,
                requested,
            } => write!(
                formatter,
                "symbolic dimension {} requests {requested}; its compile-time maximum is {max}",
                dim.0
            ),
            Self::UnsupportedOp { node, op } => {
                write!(formatter, "node {node} requires unsupported operator {op:?}")
            }
            Self::CapabilityMismatch {
                constraint,
                limit,
                actual,
            } => write!(
                formatter,
                "capability constraint `{constraint}` accepts at most {limit}, but the graph requires {actual}"
            ),
            Self::StateEffectConflict {
                state,
                first,
                second,
            } => write!(
                formatter,
                "state {} is written by node {first} and node {second}, which reverses program order",
                state.0
            ),
            Self::PlanOverflow { required, limit } => write!(
                formatter,
                "scratchpad needs {required} bytes, which exceeds the backend limit of {limit} bytes"
            ),
            Self::ValueTooLarge {
                value,
                bytes,
                limit,
            } => write!(
                formatter,
                "value {} needs {bytes} bytes, which exceeds the backend limit of {limit} bytes",
                value.0
            ),
            Self::InvalidStateShape {
                value,
                expected,
                actual,
            } => write!(
                formatter,
                "state {} has shape {actual:?}, but its StateKind implies {expected:?}",
                value.0
            ),
            Self::Unimplemented(message) => {
                write!(formatter, "unimplemented compiler path: {message}")
            }
        }
    }
}

impl core::error::Error for CompileError {}

impl From<IrError> for CompileError {
    fn from(error: IrError) -> Self {
        match error {
            IrError::SymbolicDimExceedsMax {
                dim,
                max,
                requested,
            } => Self::SymbolicDimExceedsMax {
                dim,
                max,
                requested,
            },
            other => Self::Ir(other),
        }
    }
}
