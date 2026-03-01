//! Strategy and method enums for optimization operations.

use anyhow::{bail, Result};

#[derive(Debug, Clone, Copy)]
pub(crate) enum DistillStrategy {
    Standard,
    Progressive,
    Ensemble,
}

impl std::fmt::Display for DistillStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Standard => write!(f, "standard"),
            Self::Progressive => write!(f, "progressive"),
            Self::Ensemble => write!(f, "ensemble"),
        }
    }
}

impl DistillStrategy {
    pub(super) fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "standard" | "kl" => Ok(Self::Standard),
            "progressive" | "curriculum" => Ok(Self::Progressive),
            "ensemble" => Ok(Self::Ensemble),
            _ => bail!("Unknown distill strategy: {s}. Use standard, progressive, or ensemble"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum MergeStrategy {
    Slerp,
    Ties,
    Dare,
    LinearAvg,
}

impl std::fmt::Display for MergeStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Slerp => write!(f, "slerp"),
            Self::Ties => write!(f, "ties"),
            Self::Dare => write!(f, "dare"),
            Self::LinearAvg => write!(f, "linear"),
        }
    }
}

impl MergeStrategy {
    pub(super) fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "slerp" => Ok(Self::Slerp),
            "ties" | "ties-merging" => Ok(Self::Ties),
            "dare" | "dare-ties" => Ok(Self::Dare),
            "linear" | "avg" | "linear-avg" | "average" => Ok(Self::LinearAvg),
            _ => bail!("Unknown merge strategy: {s}. Use slerp, ties, dare, or linear"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum PruneMethod {
    Wanda,
    Magnitude,
    SparseGpt,
    Structured,
    Depth,
    Width,
}

impl std::fmt::Display for PruneMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Wanda => write!(f, "wanda"),
            Self::Magnitude => write!(f, "magnitude"),
            Self::SparseGpt => write!(f, "sparsegpt"),
            Self::Structured => write!(f, "structured"),
            Self::Depth => write!(f, "depth"),
            Self::Width => write!(f, "width"),
        }
    }
}

impl PruneMethod {
    pub(super) fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "wanda" => Ok(Self::Wanda),
            "magnitude" | "mag" => Ok(Self::Magnitude),
            "sparsegpt" | "sparse-gpt" => Ok(Self::SparseGpt),
            "structured" => Ok(Self::Structured),
            "depth" => Ok(Self::Depth),
            "width" => Ok(Self::Width),
            _ => bail!("Unknown prune method: {s}. Use wanda, magnitude, sparsegpt, structured, depth, or width"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum QuantScheme {
    Int4,
    Int8,
    Q4K,
    Q5K,
    Q6K,
}

impl std::fmt::Display for QuantScheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Int4 => write!(f, "int4"),
            Self::Int8 => write!(f, "int8"),
            Self::Q4K => write!(f, "q4k"),
            Self::Q5K => write!(f, "q5k"),
            Self::Q6K => write!(f, "q6k"),
        }
    }
}

impl QuantScheme {
    pub(super) fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "int4" | "q4" => Ok(Self::Int4),
            "int8" | "q8" => Ok(Self::Int8),
            "q4k" | "q4_k" => Ok(Self::Q4K),
            "q5k" | "q5_k" => Ok(Self::Q5K),
            "q6k" | "q6_k" => Ok(Self::Q6K),
            _ => bail!("Unknown quant scheme: {s}. Use int4, int8, q4k, q5k, or q6k"),
        }
    }
}

/// HPO strategy for hyperparameter tuning.
#[derive(Debug, Clone, Copy)]
pub(crate) enum TuneStrategy {
    Tpe,
    Grid,
    Random,
}

impl std::fmt::Display for TuneStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tpe => write!(f, "tpe"),
            Self::Grid => write!(f, "grid"),
            Self::Random => write!(f, "random"),
        }
    }
}

impl TuneStrategy {
    pub(super) fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "tpe" | "bayesian" => Ok(Self::Tpe),
            "grid" | "grid-search" => Ok(Self::Grid),
            "random" | "rand" => Ok(Self::Random),
            _ => bail!("Unknown HPO strategy: {s}. Use tpe, grid, or random"),
        }
    }
}
