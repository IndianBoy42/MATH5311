use ndarray::{azip, par_azip, s, Array, Array1, Axis, IntoNdProducer};

use std::ops::{DivAssign, MulAssign, SubAssign};

pub fn gauss_elim_rust(N: usize, M: usize, A: &mut [f32], B: &mut [f32]) {
    assert_eq!(A.len(), N * N);
    assert_eq!(B.len(), N * M);
    for k in 0..(N - 1) {
        let akk = A[k * N + k];

        let mut ablk = A[(k * N)..].chunks_exact_mut(N).map(|row| &mut row[k..]);

        let arow = ablk.next().unwrap();
        let arow = &arow[1..];
        ablk.for_each(|row| {
            let (ac, lhs) = row.split_first_mut().unwrap();
            *ac /= akk;
            let ac = *ac;
            lhs.into_iter().zip(arow).for_each(|(a, ar)| {
                *a -= ac * ar;
            })
        });

        let acol = A[((k + 1) * N + k)..].iter().step_by(N);
        let mut bblk = B[(k * M)..].chunks_exact_mut(M);
        let bk = bblk.next().unwrap();
        let bk = bk.as_ref();
        bblk.zip(acol).for_each(|(bout, &a)| {
            bout.into_iter().zip(bk).for_each(|(x, b)| {
                *x -= a * b;
            })
        });
    }

    for k in (0..N).rev() {
        let akk = A[k * N + k];

        let ak = &A[(k * N + (k + 1))..((k + 1) * N)];
        let mut bblk = B[(k * N)..].chunks_exact_mut(M);
        let bk = bblk.next().unwrap();
        bblk.zip(ak).for_each(|(bk1, a)| {
            let bk1 = &*bk1;
            bk.iter_mut().zip(bk1).for_each(|(bout, &bin)| {
                *bout -= bin * a;
            })
        });

        bk.iter_mut().for_each(|x| *x /= akk);

    }
}

pub fn gauss_elim(mut A: ndarray::ArrayViewMut2<f32>, mut B: ndarray::ArrayViewMut2<f32>) {
    assert_eq!(A.nrows(), B.nrows());
    assert_eq!(A.nrows(), A.ncols());

    let N = A.nrows();
    let M = B.ncols();
    for k in 0..N {
        let a = A[(k, k)];
        let (mut a_blk, mut a_col, a_row) = A.multi_slice_mut((
            s![(k + 1).., (k + 1)..], //.
            s![(k + 1).., k],         //.
            s![k, (k + 1)..],
        ));
        let a_row = a_row.view();

        let (bk, mut brest) = B.multi_slice_mut((s![k, ..], s![(k + 1).., ..]));
        let bk = bk.view();

        azip!((mut lrow in a_blk.genrows_mut(), r in a_col, lhs in brest.genrows_mut()) {
            *r/=a;
            let r = *r;
            azip!((l in lrow, &c in a_row) {
                *l -= r * c;
            });
            azip!((lhs in lhs, &b in bk) {
                *lhs -= r * b;
            });
        });
    }

    for k in (0..N).rev() {
        let a = A.slice(s![k, (k + 1)..]);
        let (b_nxt, mut b_row) = B.multi_slice_mut((s![(k + 1).., ..], s![k, ..]));
        let akk = A[(k, k)];

        let c = a.dot(&b_nxt);
        azip!((bk in b_row, &ab in &c) {
            *bk = (*bk - ab) / akk;
        });
        // azip!((bcol in b_nxt.gencolumns(), bk in b_row) {
        //     const UNROLL: usize = 32;
        //     let ab: f32 = bcol.axis_chunks_iter(Axis(0), UNROLL)
        //         .zip(a.axis_chunks_iter(Axis(0), UNROLL))
        //         .map(|(a,b)|a.dot(&b)).sum();
        //         // .fold(0f32, |acc, (a,b)| acc + a.dot(&b));
        //         // .fold(0f32, |acc, (a,b)| acc + (a.to_owned()*b).sum());
        //     // let ab = bcol.dot(&a);
        //     *bk = (*bk - ab) / akk;
        // });
    }
}
