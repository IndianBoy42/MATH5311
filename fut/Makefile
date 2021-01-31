check: example.fut
	futhark check example.fut

example_c: example.fut
	futhark c example.fut -o example_c
example_multicore: example.fut
	futhark multicore --num-threads=8 example.fut -o example_c
libexample_c: example.fut
	futhark c --library example.fut -o example_c
libexample_multicore: example.fut
	futhark multicore --num-threads=8 --library example.fut -o example_c