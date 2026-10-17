// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SeekStarAI

//! Static graph compilation and memory layout generation for StarLight.
//!
//! [`compile`] turns a symbolic [`sl_ir::Program`] into a physical
//! [`sl_ir::CommandBuffer`] using the fixed pipeline
//! `validate → specialize → capability → decompose → effects →
//! liveness/coloring → emit`.

#![deny(unsafe_code)]

mod compile;
mod error;

pub use compile::compile;
pub use error::CompileError;
