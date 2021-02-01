use ndarray::{Axis, IntoNdProducer, azip, par_azip, s};

use std::ops::{DivAssign, SubAssign, MulAssign};

pub fn gauss_elim(mut A: ndarray::ArrayViewMut2<f32>, mut B: ndarray::ArrayViewMut2<f32>) {
    assert_eq!(A.nrows(), B.nrows());
    assert_eq!(A.nrows(), A.ncols());

    let N = A.nrows();
    let M = B.ncols();
    for k in 0..N {
        let a = A[(k, k)];
        let (mut a_blk, mut a_col, a_row) = A.multi_slice_mut((
            s![(k + 1).., (k + 1)..],
            s![(k + 1).., k],
            s![k, (k + 1)..],
        ));
        let a_row = a_row.view();

        let (bk, mut brest) = B.multi_slice_mut((s![k, ..], s![(k + 1).., ..]));
        let bk = bk.view();

        par_azip!((mut lrow in a_blk.genrows_mut(), r in a_col, lhs in brest.genrows_mut()) {
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

        par_azip!((bcol in b_nxt.gencolumns(), bk in b_row) {
            let ab = bcol.dot(&a);
            *bk = (*bk - ab) / akk;
        })
    }
}
