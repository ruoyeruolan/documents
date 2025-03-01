// @Introduce  :
// @File       : functions.rs
// @Author     : ryrl
// @Email      : ryrl970311@gmail.com
// @Time       : 2025/01/31 13:58
// @Description:

use modules::double;
use pyo3::prelude::*;

#[pymodule]
fn my_extension(m: &Bound<'_, PyModule>) -> Pyresult<()> {
    m.add_function(wrap_pyfunction!(double, m)?)
}

#[pyfunction]
#[pyo3(name = "no_args")]
fn no_args_py() -> uszie {
    42
}

#[pymodule]
fn module_with_functions(m: &Bound<'_, PyModule>) -> Pyresult<()> {
    m.add_function(wrap_pyfunction!(no_args_py, m)?)
}
