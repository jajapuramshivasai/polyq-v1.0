//! A small OpenQASM-2-style reader/writer for the supported gate set.
//!
//! Recognised statements (case-sensitive): `qreg q[n];`, `h x y z s sdg t tdg`,
//! `p(expr)` / `u1(expr)` / `rz(expr)`, `cz cx swap ccx ccz`.  `rz(θ)` follows the
//! Qiskit convention (global phase `e^{-iθ/2}` is tracked).  `OPENQASM`, `include`,
//! `creg`, `barrier`, `measure` and `//` comments are ignored.  Anything else is an
//! error (silently dropping a gate would silently change the circuit).
//!
//! Angle expressions support numbers, `pi`, `+ - * /`, unary minus and parentheses.

use crate::circuit::{Circuit, Gate};
use crate::phase::Phase;
use std::fmt;
use std::fs;
use std::io;

/// QASM I/O error.
#[derive(Debug)]
pub enum QasmError {
    /// Underlying I/O failure.
    Io(io::Error),
    /// Syntax or semantic error with a message.
    Parse(String),
}

impl fmt::Display for QasmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QasmError::Io(e) => write!(f, "io error: {e}"),
            QasmError::Parse(s) => write!(f, "qasm parse error: {s}"),
        }
    }
}
impl std::error::Error for QasmError {}
impl From<io::Error> for QasmError {
    fn from(e: io::Error) -> Self {
        QasmError::Io(e)
    }
}

fn perr<T>(s: impl Into<String>) -> Result<T, QasmError> {
    Err(QasmError::Parse(s.into()))
}

// ---- tiny expression evaluator -------------------------------------------------

struct Expr<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> Expr<'a> {
    fn ws(&mut self) {
        while self.i < self.s.len() && (self.s[self.i] as char).is_whitespace() {
            self.i += 1;
        }
    }
    fn eat(&mut self, c: u8) -> bool {
        self.ws();
        if self.i < self.s.len() && self.s[self.i] == c {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn expr(&mut self) -> Result<f64, QasmError> {
        let mut v = self.term()?;
        loop {
            if self.eat(b'+') {
                v += self.term()?;
            } else if self.eat(b'-') {
                v -= self.term()?;
            } else {
                return Ok(v);
            }
        }
    }
    fn term(&mut self) -> Result<f64, QasmError> {
        let mut v = self.unary()?;
        loop {
            if self.eat(b'*') {
                v *= self.unary()?;
            } else if self.eat(b'/') {
                v /= self.unary()?;
            } else {
                return Ok(v);
            }
        }
    }
    fn unary(&mut self) -> Result<f64, QasmError> {
        if self.eat(b'-') {
            return Ok(-self.unary()?);
        }
        if self.eat(b'+') {
            return self.unary();
        }
        self.atom()
    }
    fn atom(&mut self) -> Result<f64, QasmError> {
        self.ws();
        if self.eat(b'(') {
            let v = self.expr()?;
            if !self.eat(b')') {
                return perr("missing `)` in angle expression");
            }
            return Ok(v);
        }
        let st = self.i;
        if self.s[self.i..].starts_with(b"pi") {
            self.i += 2;
            return Ok(std::f64::consts::PI);
        }
        while self.i < self.s.len() && (self.s[self.i].is_ascii_digit() || matches!(self.s[self.i], b'.' | b'e' | b'E')) {
            // allow exponent sign directly after e/E
            if matches!(self.s[self.i], b'e' | b'E') && self.i + 1 < self.s.len() && matches!(self.s[self.i + 1], b'+' | b'-') {
                self.i += 1;
            }
            self.i += 1;
        }
        std::str::from_utf8(&self.s[st..self.i])
            .ok()
            .and_then(|t| t.parse::<f64>().ok())
            .map_or_else(|| perr(format!("bad number in angle expression near byte {st}")), Ok)
    }
}

/// Evaluate an angle expression such as `pi/4`, `-3*pi/8` or `0.125`.
pub fn eval_angle(src: &str) -> Result<f64, QasmError> {
    let mut e = Expr { s: src.as_bytes(), i: 0 };
    let v = e.expr()?;
    e.ws();
    if e.i != e.s.len() {
        return perr(format!("trailing characters in angle `{src}`"));
    }
    Ok(v)
}

fn qubit(tok: &str) -> Result<usize, QasmError> {
    let (a, b) = (tok.find('['), tok.find(']'));
    match (a, b) {
        (Some(a), Some(b)) if a < b => tok[a + 1..b].trim().parse().map_err(|_| QasmError::Parse(format!("bad qubit index in `{tok}`"))),
        _ => perr(format!("bad qubit operand `{tok}`")),
    }
}

/// Parse a QASM string.
pub fn parse_qasm(src: &str) -> Result<Circuit, QasmError> {
    let mut c = Circuit::default();
    let cleaned: String = src.lines().map(|l| l.split("//").next().unwrap_or("")).collect::<Vec<_>>().join("\n");
    for stmt in cleaned.split(';') {
        let stmt = stmt.trim();
        if stmt.is_empty() {
            continue;
        }
        // split into name(+args) and operands
        let (head, rest) = match stmt.find(|ch: char| ch.is_whitespace()) {
            Some(_) if !stmt.contains('(') => {
                let k = stmt.find(char::is_whitespace).unwrap();
                (&stmt[..k], stmt[k..].trim())
            }
            _ if stmt.contains('(') => {
                let close = stmt.rfind(')').ok_or_else(|| QasmError::Parse(format!("unbalanced `(` in `{stmt}`")))?;
                (&stmt[..=close], stmt[close + 1..].trim())
            }
            _ => (stmt, ""),
        };
        let (name, arg) = match head.find('(') {
            Some(k) => (head[..k].trim(), Some(&head[k + 1..head.len() - 1])),
            None => (head.trim(), None),
        };
        let ops: Vec<&str> = rest.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
        match (name, ops.as_slice()) {
            ("OPENQASM", _) | ("include", _) | ("creg", _) | ("barrier", _) | ("measure", _) => {}
            ("qreg", [q]) => c.num_qubits = c.num_qubits.max(qubit(q)? + 0).max(parse_qreg_size(q)?),
            ("h", [q]) => {
                c.h(qubit(q)?);
            }
            ("x", [q]) => {
                c.x(qubit(q)?);
            }
            ("y", [q]) => {
                c.y(qubit(q)?);
            }
            ("z", [q]) => {
                c.z(qubit(q)?);
            }
            ("s", [q]) => {
                c.s(qubit(q)?);
            }
            ("sdg", [q]) => {
                c.sdg(qubit(q)?);
            }
            ("t", [q]) => {
                c.t(qubit(q)?);
            }
            ("tdg", [q]) => {
                c.tdg(qubit(q)?);
            }
            ("p", [q]) | ("u1", [q]) => {
                let th = eval_angle(arg.ok_or_else(|| QasmError::Parse("p needs an angle".into()))?)?;
                c.p(qubit(q)?, Phase::from_radians(th));
            }
            ("rz", [q]) => {
                let th = eval_angle(arg.ok_or_else(|| QasmError::Parse("rz needs an angle".into()))?)?;
                c.rz(qubit(q)?, th);
            }
            ("gphase", []) => {
                let th = eval_angle(arg.ok_or_else(|| QasmError::Parse("gphase needs an angle".into()))?)?;
                c.gphase(Phase::from_radians(th));
            }
            ("cz", [a, b]) => {
                c.cz(qubit(a)?, qubit(b)?);
            }
            ("cx", [a, b]) | ("CX", [a, b]) => {
                c.cx(qubit(a)?, qubit(b)?);
            }
            ("swap", [a, b]) => {
                c.swap(qubit(a)?, qubit(b)?);
            }
            ("ccx", [a, b, t]) => {
                c.ccx(qubit(a)?, qubit(b)?, qubit(t)?);
            }
            ("ccz", [a, b, t]) => {
                c.ccz(qubit(a)?, qubit(b)?, qubit(t)?);
            }
            _ => return perr(format!("unsupported statement `{stmt}`")),
        }
    }
    Ok(c)
}

fn parse_qreg_size(tok: &str) -> Result<usize, QasmError> {
    // `q[5]` declares 5 qubits (index == size here)
    qubit(tok)
}

/// Read a QASM file.
pub fn read_qasm_file(path: &str) -> Result<Circuit, QasmError> {
    parse_qasm(&fs::read_to_string(path)?)
}

/// Render a circuit as QASM.  Rotation angles are written as exact rationals of π
/// (`pi*k/2147483648`) so that `parse_qasm(write_qasm(c)) == c` for [`Phase`] angles.
pub fn write_qasm_string(c: &Circuit) -> String {
    let mut s = format!("OPENQASM 2.0;\nqreg q[{}];\n", c.num_qubits);
    let ang = |p: Phase| format!("pi*{}/2147483648", p.0);
    for g in &c.gates {
        match *g {
            Gate::H(q) => s += &format!("h q[{q}];\n"),
            Gate::X(q) => s += &format!("x q[{q}];\n"),
            Gate::Y(q) => s += &format!("y q[{q}];\n"),
            Gate::Z(q) => s += &format!("z q[{q}];\n"),
            Gate::S(q) => s += &format!("s q[{q}];\n"),
            Gate::Sdg(q) => s += &format!("sdg q[{q}];\n"),
            Gate::T(q) => s += &format!("t q[{q}];\n"),
            Gate::Tdg(q) => s += &format!("tdg q[{q}];\n"),
            Gate::P(q, p) => s += &format!("p({}) q[{q}];\n", ang(p)),
            Gate::CZ(a, b) => s += &format!("cz q[{a}],q[{b}];\n"),
            Gate::CX(a, b) => s += &format!("cx q[{a}],q[{b}];\n"),
            Gate::Swap(a, b) => s += &format!("swap q[{a}],q[{b}];\n"),
            Gate::GPhase(p) => s += &format!("gphase({});\n", ang(p)),
        }
    }
    s
}

/// Write a circuit to a QASM file.
pub fn write_qasm_file(path: &str, c: &Circuit) -> io::Result<()> {
    fs::write(path, write_qasm_string(c))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angles() {
        assert!((eval_angle("pi/4").unwrap() - std::f64::consts::FRAC_PI_4).abs() < 1e-15);
        assert!((eval_angle("-3*pi/8").unwrap() + 3.0 * std::f64::consts::PI / 8.0).abs() < 1e-15);
        assert!((eval_angle("(1+1)*0.5").unwrap() - 1.0).abs() < 1e-15);
        assert!(eval_angle("pi/").is_err());
    }

    #[test]
    fn roundtrip() {
        let mut c = Circuit::new(3);
        c.h(0).t(1).tdg(2).cx(0, 1).cz(1, 2).swap(0, 2).s(0).sdg(1).x(2).y(0).z(1);
        c.p(1, Phase::from_pi_fraction(5, 7));
        c.gphase(Phase(12345));
        let text = write_qasm_string(&c);
        let back = parse_qasm(&text).unwrap();
        assert_eq!(back.gates, c.gates);
        assert_eq!(back.num_qubits, 3);
    }

    #[test]
    fn rejects_unknown() {
        assert!(parse_qasm("qreg q[1]; frobnicate q[0];").is_err());
    }

    #[test]
    fn parses_qiskit_style() {
        let c = parse_qasm("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[2];\nh q[0]; cx q[0],q[1]; rz(pi/4) q[1]; // hi\n").unwrap();
        assert_eq!(c.num_qubits, 2);
        assert_eq!(c.gates.len(), 4); // h, cx, P, GPhase
    }
}
