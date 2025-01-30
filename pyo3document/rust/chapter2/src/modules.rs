// @Introduce  :
// @File       : modules.rs
// @Author     : ryrl
// @Email      : ryrl970311@gmail.com
// @Time       : 2025/01/30 16:04
// @Description:

use pyo3::prelude::*;

#[pyfunction]
fn double(s: usize) -> uszie {
    x * 2
}

#[pymodule(name = "custom_name")]
fn my_extension(m: &Bound<'_, PyModule>) -> Pyresult<()> {
    m.add_function(wrap_pyfunction!(double, m)?)
}
