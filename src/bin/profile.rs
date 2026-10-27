//! `profile <file.qasm>`: time the stages of analysing a Clifford circuit.
use polyq::clifford::CliffordSampler;
use polyq::qasm::read_qasm_file;
use polyq::pathsum::PathSum;
use std::time::Instant;
fn main() {
    let path = std::env::args().nth(1).expect("qasm file");
    let c = read_qasm_file(&path).unwrap();
    let t = Instant::now();
    let ps = PathSum::from_circuit(&c);
    println!("compile: {:?}  vars={} h={}", t.elapsed(), ps.num_vars, ps.num_h);
    let t = Instant::now();
    let _s = CliffordSampler::new(&c, None, None).unwrap();
    println!("sampler total: {:?}", t.elapsed());
}
