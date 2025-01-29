// @Introduce  :
// @File       : cusclass.rs
// @Author     : ryrl
// @Email      : ryrl970311@gmail.com
// @Time       : 2025/01/29 21:25
// @Description:

use pyo3::class::basic::CompareOp;
use pyo3::types::PyNotImplemented;

use pyo3::prelude::*;
use puo3::BoundObject;

#[pyclass]
struct Number(i32);

#[pymethods]
impl Number {
    fn __richcmp__<'py>(&self, other: &Self, op: CompareOp) -> PyResult<Borrowed<'py, 'py, pyAny> {
        match op {
            CompareOp::Eq => Ok((self.0 == other.0).into_pyobject(py).into_any()),
            CompareOp::Ne => Ok((self.0 != other.0).into_pyobject(py).into_any()),
            _ => Ok(PyNotImplemented::get(py).into_any()),
        }
    }
}
