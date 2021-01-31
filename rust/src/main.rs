use std::error::Error;
use std::fs::File;

use csv::{ReaderBuilder, WriterBuilder};
use gauss_elim::gauss_elim;
use math5311_rust::*;
use ndarray::array;
use ndarray::{Array, Array2};
use ndarray_csv::{Array2Reader, Array2Writer};

// constexpr int N = 3;
// std::array<float, N *N> exampleA = {
// 	2,	1, -1,	//
// 	4,	5, -3,	//
// 	-2, 5, -2,	//
// };
// constexpr int N = 4;
// const std::array<float, N *N> srcMat = {
// 	3,	5, 7, 2,   //.
// 	1,	4, 7, 2,   //.
// 	6,	3, 9, 17,  //.
// 	13, 5, 4, 16,  //.
// };

fn main() {
    let f = || -> Result<_, Box<dyn Error>> {
        let f = File::open("../output.rmat")?;
        let mut reader = ReaderBuilder::new().has_headers(false).from_reader(f);
        let array: Array2<f32> = reader.deserialize_array2((1000, 1000))?;
        Ok(array)
    };

    let mut A = array![
        [2., 1., -1.,],
        [4., 5., -3.,],
        [-2., 5., -2.,],
        // [3., 5., 7., 2.,],
        // [1., 4., 7., 2.,],
        // [6., 3., 9., 17.,],
        // [13., 5., 4., 16.,],
    ];
    let mut B = A.clone();
    gauss_elim(A.view_mut(), B.view_mut());
    dbg!(B);

    let mut A = f().map_err(|x| dbg!(x)).unwrap();
    let mut B = A.clone();

    for _ in 0..10 {
        let (_, time) = tempus_fugit::measure!(gauss_elim(A.view_mut(), B.view_mut()));
        println!("{}", time);
    }
    println!("Hello, world!");
}
