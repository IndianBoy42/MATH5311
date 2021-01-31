use ndarray::{azip, par_azip, s};

use std::ops::{DivAssign, SubAssign};

pub fn gauss_elim(mut A: ndarray::ArrayViewMut2<f32>, mut B: ndarray::ArrayViewMut2<f32>) {
    assert_eq!(A.nrows(), B.nrows());
    assert_eq!(A.nrows(), A.ncols());

    let N = A.nrows();
    let M = B.ncols();
    for k in 0..N {
        let a = A[(k, k)];
        let (mut a_blk, mut a_col, a_row) =
            A.multi_slice_mut((s![(k + 1)..N, (k + 1)..N], s![(k + 1)..N, k], s![k, (k + 1)..N]));
        let a_row = a_row.view();
        
        par_azip!((lrow in a_blk.genrows_mut(), r in &mut a_col) {
            *r/=a;
            let r = *r;
            azip!((l in lrow, &c in a_row) {
                *l -= r*c;
            })
        });
        let a_col = a_col.view();

        let (bk, mut brest) = B.multi_slice_mut((s![k, 0..M], s![(k + 1)..N, 0..M]));
        let bk = bk.view();
        par_azip!((lhs in brest.genrows_mut(), &a in a_col)  {
            azip!((lhs in lhs, &b in bk) {
                *lhs -= b * a;
            })
        })
    }

    for k in (0..N).rev() {
        let a = A.slice(s![k, (k + 1)..N]);
        let (b_nxt, mut b_row) = B.multi_slice_mut((s![(k + 1)..N, ..M], s![k, 0..M]));
        let akk = A[(k, k)];

        // Incorrect
        // ndarray::Zip::from(b_nxt)
        //     .and_broadcast(b_row)
        //     .and_broadcast(a)
        //     .apply(|b, bout, &a| *bout -= a * b);

        par_azip!((bcol in b_nxt.gencolumns(), bk in b_row) {
                let ab = ndarray::Zip::from(bcol)
                    .and(a)
                    .fold(0f32, |acc, a, b| acc + a * b);
                *bk = (*bk - ab) / akk;
        })
    }
}
