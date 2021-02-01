use rayon::prelude::*;

pub fn cholesky(N: usize, M: usize, A: &mut [f32], B: &mut [f32]) {
    assert_eq!(A.len(), N * N);
    assert_ne!(N, 0);
    assert_ne!(M, 0);
    assert_eq!(B.len(), N * M);

    for j in 0..N {
        let ajp = &mut A[j * N..];
        let (aj, ajp) = ajp.split_at_mut(N);
        let (ajj, aj) = aj[..(j + 1)].split_last_mut().unwrap();
        let aj = &*aj;

        *ajj = (*ajj - aj.iter().map(|&a| a * a).sum::<f32>()).sqrt();

        let ajj = *ajj;

        ajp.chunks_exact_mut(N)
            .map(|row| &mut row[..(j+1)])
            .map(|row| row.split_last_mut().unwrap())
            .for_each(|(out, &mut ref ai)| {
                let c: f32 = aj.iter().zip(ai).map(|(&i, &j)| i * j).sum();
                *out = (*out - c) / ajj;
            });

        let acol = A[((j + 1) * N)..]
            .chunks_exact(N)
            .map(|x| unsafe { x.get_unchecked(j) });
        // let acol = A[((j + 1) * N + j)..].iter().step_by(N);

        let (bj, bblk) = B[(j * M)..].split_at_mut(M);
        let bblk = bblk.chunks_exact_mut(M);
        let bj = bj.as_ref();
        bblk.zip(acol).for_each(|(bout, &a)| {
            bout.into_iter().zip(bj).for_each(|(x, b)| {
                *x -= a * b;
            })
        });
    }
    B[N * N - 1] /= A[N * N - 1];
    for k in (0..(N - 1)).rev() {
        let akk = A[k * N + k];
        let ak = A[(k * N)..]
            .chunks_exact(N)
            .map(|x| unsafe { x.get_unchecked(k) });

        let (bk, bblk) = B[(k * M)..].split_at_mut(M);
        let bblk = bblk.chunks_exact(M);
        bblk.zip(ak).for_each(|(bk1, &a)| {
            bk.iter_mut().zip(bk1).for_each(|(bout, &bin)| {
                *bout -= bin * a;
            })
        });
        bk.iter_mut().for_each(|x| *x /= akk);
    }
}
