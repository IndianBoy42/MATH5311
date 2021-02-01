use ndarray::{azip, par_azip, s, Array, Array1, Axis, IntoNdProducer};

use std::ops::{DivAssign, MulAssign, SubAssign};

pub fn gauss_elim(mut A: ndarray::ArrayViewMut2<f32>, mut B: ndarray::ArrayViewMut2<f32>) {
    assert_eq!(A.nrows(), B.nrows());
    assert_eq!(A.nrows(), A.ncols());

    let N = A.nrows();
    let M = B.ncols();
    for k in 0..N {
        let a = A[(k, k)];
        let (mut a_blk, mut a_col, a_row) = A.multi_slice_mut((
            s![(k + 1).., (k + 1)..],  //.
            s![(k + 1).., k..(k + 1)], //.
            s![k..(k + 1), (k + 1)..],
        ));
        let a_row = a_row.view();

        let (bk, mut brest) = B.multi_slice_mut((
            s![k..(k + 1), ..], //.
            s![(k + 1).., ..],
        ));
        let bk = bk.view();

        a_col.div_assign(a);
        let c = a_col.dot(&a_row);
        a_blk.sub_assign(&c);
        // azip!((aij in a_blk, cij in &c) {
        //     *aij -= cij;
        // });
        let ab = a_col.dot(&bk);
        brest.sub_assign(&ab);
        // azip!((b in brest, &abij in &ab) {
        //     *b -= abij;
        // });
        // azip!((mut lrow in a_blk.genrows_mut(), r in a_col, lhs in brest.genrows_mut()) {
        //     *r/=a;
        //     let r = *r;
        //     azip!((l in lrow, &c in a_row) {
        //         *l -= r * c;
        //     });
        //     azip!((lhs in lhs, &b in bk) {
        //         *lhs -= r * b;
        //     });
        // });
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