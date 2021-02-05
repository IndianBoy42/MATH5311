use std::error::Error;
use std::fs::File;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use csv::ReaderBuilder;
use math5311_rust::cholesky::cholesky;
use math5311_rust::gauss_elim::{gauss_elim, gauss_elim_rust};
use ndarray::Array2;
use ndarray_csv::Array2Reader;

pub fn criterion_benchmark(c: &mut Criterion) {
    let f = || -> Result<_, Box<dyn Error>> {
        let f = File::open("../output.rmat")?;
        let mut reader = ReaderBuilder::new().has_headers(false).from_reader(f);
        let array: Array2<f32> = reader.deserialize_array2((1000, 1000))?;
        Ok(array)
    };
    let A = f().map_err(|x| dbg!(x)).unwrap();
    let N = A.ncols();

    c.bench_function("cholesky", |b| {
        b.iter(|| {
            let mut A = A.clone();
            let mut B = A.clone();
            let A = A.as_slice_mut().unwrap();
            let B = B.as_slice_mut().unwrap();
            cholesky(N, N, A, B);
        })
    });
    c.bench_function("gauss_elim", |b| {
        b.iter(|| {
            let mut A = A.clone();
            let mut B = A.clone();
            let A = A.as_slice_mut().unwrap();
            let B = B.as_slice_mut().unwrap();
            gauss_elim_rust(N, N, A, B);
        })
    });

    c.bench_function("gauss_elim_nd", |b| {
        b.iter(|| {
            let mut A = A.clone();
            let mut B = A.clone();
            gauss_elim(A.view_mut(), B.view_mut());
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
