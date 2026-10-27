//! The dyadic phase ring `Z_{2^32}` and Z4 helpers.
//!
//! A [`Phase`] stores an integer `k` meaning the angle `θ = 2π · k / 2^32`.
//! Addition is wrapping `u32` addition, which is **exactly** addition of angles
//! modulo `2π`.
//!
//! # Why 2π and not π
//!
//! The v1 crate stored phases in units of `π / 2^16` with `u16` wrap-around.
//! That ring has modulus `π` (not `2π`), so `-1 = e^{iπ}` aliased to `+1`
//! (and `T†` was unrepresentable).  The ring here has modulus `2π`, so
//! `Z = Phase(2^31)`, `S = Phase(2^30)`, `T = Phase(2^29)`, `T† = -T`.
//!
//! Any angle that is a dyadic multiple `π · j / 2^k` with `k <= 31` is exact;
//! other angles are rounded to the nearest of the `2^32` grid points
//! (error `<= π / 2^32 ≈ 7.3e-10` rad).

use num_complex::Complex64;
use std::f64::consts::PI;
use std::ops::{Add, AddAssign, Neg, Sub, SubAssign};

/// An angle in units of `2π / 2^32`.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct Phase(pub u32);

impl Phase {
    /// `θ = 0`.
    pub const ZERO: Phase = Phase(0);
    /// `θ = π` (the `Z` gate).
    pub const PI: Phase = Phase(1 << 31);
    /// `θ = π/2` (the `S` gate).
    pub const HALF_PI: Phase = Phase(1 << 30);
    /// `θ = π/4` (the `T` gate).
    pub const QUARTER_PI: Phase = Phase(1 << 29);
    /// Number of grid points on the circle.
    pub const MODULUS: f64 = 4294967296.0;

    /// Round `radians` to the nearest grid point.
    pub fn from_radians(radians: f64) -> Phase {
        let turns = radians / (2.0 * PI);
        let frac = turns - turns.floor();
        let k = (frac * Self::MODULUS).round();
        Phase((k as u64 & 0xFFFF_FFFF) as u32)
    }

    /// The exact angle `π · num / 2^log2_den` (requires `log2_den <= 31`).
    pub fn from_pi_fraction(num: i64, log2_den: u32) -> Phase {
        assert!(log2_den <= 31, "denominator too fine for the 2^32 phase ring");
        // π·num/2^e = 2π · num / 2^(e+1) = 2π · num · 2^(31-e) / 2^32
        let k = (num as i128) << (31 - log2_den);
        Phase((k.rem_euclid(1i128 << 32)) as u32)
    }

    /// Angle in `[0, 2π)`.
    pub fn to_radians(self) -> f64 {
        2.0 * PI * (self.0 as f64) / Self::MODULUS
    }

    /// `e^{iθ}`.
    pub fn to_complex(self) -> Complex64 {
        // Use exact values on the 8th roots to avoid sin/cos noise.
        if self.0 & ((1 << 29) - 1) == 0 {
            return ROOT8[(self.0 >> 29) as usize];
        }
        Complex64::from_polar(1.0, self.to_radians())
    }

    /// `Some(k)` iff `θ = k·π/2` for `k in 0..4` (i.e. the rotation is Clifford).
    #[inline(always)]
    pub fn clifford_level(self) -> Option<u8> {
        if self.0 & ((1 << 30) - 1) == 0 {
            Some((self.0 >> 30) as u8)
        } else {
            None
        }
    }

    /// `true` iff `θ` is an odd multiple of `π/4` ("T-class" rotation).
    #[inline(always)]
    pub fn is_t_class(self) -> bool {
        self.0 & ((1 << 29) - 1) == 0 && (self.0 >> 29) & 1 == 1
    }

    /// Exponent of the 8th root of unity if `θ = k·π/4`.
    #[inline(always)]
    pub fn eighth_root_index(self) -> Option<u8> {
        if self.0 & ((1 << 29) - 1) == 0 {
            Some((self.0 >> 29) as u8)
        } else {
            None
        }
    }
}

/// `ROOT8[k] = e^{iπk/4}`, written out exactly.
pub const ROOT8: [Complex64; 8] = {
    const H: f64 = std::f64::consts::FRAC_1_SQRT_2;
    [
        Complex64::new(1.0, 0.0),
        Complex64::new(H, H),
        Complex64::new(0.0, 1.0),
        Complex64::new(-H, H),
        Complex64::new(-1.0, 0.0),
        Complex64::new(-H, -H),
        Complex64::new(0.0, -1.0),
        Complex64::new(H, -H),
    ]
};

/// The phase `k · π/2` for `k` in `Z4` (any integer, reduced mod 4).
#[inline(always)]
pub fn z4_phase(k: u8) -> Phase {
    Phase(((k & 3) as u32) << 30)
}

impl Add for Phase {
    type Output = Phase;
    #[inline(always)]
    fn add(self, o: Phase) -> Phase {
        Phase(self.0.wrapping_add(o.0))
    }
}
impl Sub for Phase {
    type Output = Phase;
    #[inline(always)]
    fn sub(self, o: Phase) -> Phase {
        Phase(self.0.wrapping_sub(o.0))
    }
}
impl Neg for Phase {
    type Output = Phase;
    #[inline(always)]
    fn neg(self) -> Phase {
        Phase(self.0.wrapping_neg())
    }
}
impl AddAssign for Phase {
    #[inline(always)]
    fn add_assign(&mut self, o: Phase) {
        self.0 = self.0.wrapping_add(o.0);
    }
}
impl SubAssign for Phase {
    #[inline(always)]
    fn sub_assign(&mut self, o: Phase) {
        self.0 = self.0.wrapping_sub(o.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants() {
        assert_eq!(Phase::QUARTER_PI + Phase::QUARTER_PI, Phase::HALF_PI);
        assert_eq!(Phase::HALF_PI + Phase::HALF_PI, Phase::PI);
        assert_eq!(Phase::PI + Phase::PI, Phase::ZERO);
        assert_eq!(-Phase::QUARTER_PI + Phase::QUARTER_PI, Phase::ZERO);
        assert!((Phase::PI.to_complex() + Complex64::new(1.0, 0.0)).norm() < 1e-15);
    }

    #[test]
    fn radians_roundtrip() {
        for k in [0u32, 1, 12345, 1 << 29, (1 << 31) + 7, u32::MAX] {
            let p = Phase(k);
            assert_eq!(Phase::from_radians(p.to_radians()), p);
        }
        assert_eq!(Phase::from_radians(-PI / 4.0), -Phase::QUARTER_PI);
        assert_eq!(Phase::from_pi_fraction(-1, 2), -Phase::QUARTER_PI);
        assert_eq!(Phase::from_pi_fraction(3, 2).clifford_level(), None);
        assert!(Phase::from_pi_fraction(3, 2).is_t_class());
    }

    #[test]
    fn clifford_levels() {
        assert_eq!(Phase::ZERO.clifford_level(), Some(0));
        assert_eq!(Phase::HALF_PI.clifford_level(), Some(1));
        assert_eq!(Phase::PI.clifford_level(), Some(2));
        assert_eq!((-Phase::HALF_PI).clifford_level(), Some(3));
        assert_eq!(Phase::QUARTER_PI.clifford_level(), None);
    }
}
