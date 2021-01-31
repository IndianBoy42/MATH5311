use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("Hello World", |b| {
        b.iter(|| black_box(0))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);