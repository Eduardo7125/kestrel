//! The ggml tensor type table: block size and bytes per block for every type a
//! GGUF file can carry. Sizes mirror `ggml/src/ggml-common.h` (`static_assert`s
//! on the `block_*` structs) and `type_traits` in `ggml/src/ggml.c`.

use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1_M,
    BF16,
    TQ1_0,
    TQ2_0,
    MXFP4,
    NVFP4,
    Q1_0,
    Q2_0,
}

impl GgmlType {
    pub fn from_id(id: u32) -> Option<Self> {
        use GgmlType::*;
        Some(match id {
            0 => F32,
            1 => F16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            9 => Q8_1,
            10 => Q2_K,
            11 => Q3_K,
            12 => Q4_K,
            13 => Q5_K,
            14 => Q6_K,
            15 => Q8_K,
            16 => IQ2_XXS,
            17 => IQ2_XS,
            18 => IQ3_XXS,
            19 => IQ1_S,
            20 => IQ4_NL,
            21 => IQ3_S,
            22 => IQ2_S,
            23 => IQ4_XS,
            24 => I8,
            25 => I16,
            26 => I32,
            27 => I64,
            28 => F64,
            29 => IQ1_M,
            30 => BF16,
            34 => TQ1_0,
            35 => TQ2_0,
            39 => MXFP4,
            40 => NVFP4,
            41 => Q1_0,
            42 => Q2_0,
            _ => return None,
        })
    }

    pub fn id(self) -> u32 {
        use GgmlType::*;
        match self {
            F32 => 0,
            F16 => 1,
            Q4_0 => 2,
            Q4_1 => 3,
            Q5_0 => 6,
            Q5_1 => 7,
            Q8_0 => 8,
            Q8_1 => 9,
            Q2_K => 10,
            Q3_K => 11,
            Q4_K => 12,
            Q5_K => 13,
            Q6_K => 14,
            Q8_K => 15,
            IQ2_XXS => 16,
            IQ2_XS => 17,
            IQ3_XXS => 18,
            IQ1_S => 19,
            IQ4_NL => 20,
            IQ3_S => 21,
            IQ2_S => 22,
            IQ4_XS => 23,
            I8 => 24,
            I16 => 25,
            I32 => 26,
            I64 => 27,
            F64 => 28,
            IQ1_M => 29,
            BF16 => 30,
            TQ1_0 => 34,
            TQ2_0 => 35,
            MXFP4 => 39,
            NVFP4 => 40,
            Q1_0 => 41,
            Q2_0 => 42,
        }
    }

    /// Elements per block.
    pub fn block_size(self) -> usize {
        use GgmlType::*;
        match self {
            F32 | F16 | BF16 | I8 | I16 | I32 | I64 | F64 => 1,
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1 | IQ4_NL | MXFP4 => 32,
            NVFP4 | Q2_0 => 64,
            Q1_0 => 128,
            Q2_K | Q3_K | Q4_K | Q5_K | Q6_K | Q8_K | IQ2_XXS | IQ2_XS | IQ3_XXS | IQ1_S
            | IQ3_S | IQ2_S | IQ4_XS | IQ1_M | TQ1_0 | TQ2_0 => 256,
        }
    }

    /// Bytes per block.
    pub fn type_size(self) -> usize {
        use GgmlType::*;
        match self {
            F32 => 4,
            F16 | BF16 => 2,
            I8 => 1,
            I16 => 2,
            I32 => 4,
            I64 | F64 => 8,
            Q4_0 => 18,
            Q4_1 => 20,
            Q5_0 => 22,
            Q5_1 => 24,
            Q8_0 => 34,
            Q8_1 => 36,
            Q2_K => 84,
            Q3_K => 110,
            Q4_K => 144,
            Q5_K => 176,
            Q6_K => 210,
            Q8_K => 292,
            IQ2_XXS => 66,
            IQ2_XS => 74,
            IQ3_XXS => 98,
            IQ1_S => 50,
            IQ4_NL => 18,
            IQ3_S => 110,
            IQ2_S => 82,
            IQ4_XS => 136,
            IQ1_M => 56,
            TQ1_0 => 54,
            TQ2_0 => 66,
            MXFP4 => 17,
            NVFP4 => 36,
            Q1_0 => 18,
            Q2_0 => 18,
        }
    }

    /// Bytes needed to store `n` elements (must be a multiple of the block size).
    pub fn bytes_for(self, n: u64) -> Option<u64> {
        let bs = self.block_size() as u64;
        if n % bs != 0 {
            return None;
        }
        Some(n / bs * self.type_size() as u64)
    }

    /// Average bits per weight, including block scales.
    pub fn bits_per_weight(self) -> f64 {
        self.type_size() as f64 * 8.0 / self.block_size() as f64
    }

    pub fn name(self) -> &'static str {
        use GgmlType::*;
        match self {
            F32 => "F32",
            F16 => "F16",
            Q4_0 => "Q4_0",
            Q4_1 => "Q4_1",
            Q5_0 => "Q5_0",
            Q5_1 => "Q5_1",
            Q8_0 => "Q8_0",
            Q8_1 => "Q8_1",
            Q2_K => "Q2_K",
            Q3_K => "Q3_K",
            Q4_K => "Q4_K",
            Q5_K => "Q5_K",
            Q6_K => "Q6_K",
            Q8_K => "Q8_K",
            IQ2_XXS => "IQ2_XXS",
            IQ2_XS => "IQ2_XS",
            IQ3_XXS => "IQ3_XXS",
            IQ1_S => "IQ1_S",
            IQ4_NL => "IQ4_NL",
            IQ3_S => "IQ3_S",
            IQ2_S => "IQ2_S",
            IQ4_XS => "IQ4_XS",
            I8 => "I8",
            I16 => "I16",
            I32 => "I32",
            I64 => "I64",
            F64 => "F64",
            IQ1_M => "IQ1_M",
            BF16 => "BF16",
            TQ1_0 => "TQ1_0",
            TQ2_0 => "TQ2_0",
            MXFP4 => "MXFP4",
            NVFP4 => "NVFP4",
            Q1_0 => "Q1_0",
            Q2_0 => "Q2_0",
        }
    }
}

impl std::fmt::Display for GgmlType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// `general.file_type` (llama_ftype) → human name, as llama.cpp prints it.
pub fn file_type_name(ftype: u32) -> &'static str {
    match ftype {
        0 => "all F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "MXFP4_MOE",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_ids() {
        for id in 0..64 {
            if let Some(t) = GgmlType::from_id(id) {
                assert_eq!(t.id(), id);
            }
        }
    }

    #[test]
    fn known_bpw() {
        assert_eq!(GgmlType::Q4_0.bits_per_weight(), 4.5);
        assert_eq!(GgmlType::Q8_0.bits_per_weight(), 8.5);
        assert_eq!(GgmlType::Q4_K.bits_per_weight(), 4.5);
        assert!((GgmlType::Q6_K.bits_per_weight() - 6.5625).abs() < 1e-9);
        assert_eq!(GgmlType::Q4_K.bytes_for(512), Some(288));
        assert_eq!(GgmlType::Q4_K.bytes_for(100), None);
    }
}
