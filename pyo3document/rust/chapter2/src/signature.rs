// @Introduce  :
// @File       : signature.rs
// @Author     : ryrl
// @Email      : ryrl970311@gmail.com
// @Time       : 2025/01/31 14:11
// @Description:

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

#[pyfunction]
#[pyo3(signature = (**kwds))]
fn num_kwds(kwds: Option<&Bound<'_, PyModule>>) -> usize {
    kwds.map_or(0, |dict| dict.len())
}

#[pymodule]
fn module_with_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(num_kwds, m)?)
}

#[pymethods]
impl MyClass {
    #[new]
    #[pyo3(signature = (num=-1))]
    fn new(num: i32) -> Self {
        MyClass { num }
    }

    #[pyo3(signature = (num=10, *py_args, name="Hello", **py_kwargs))]
    fn method(
        &mut self,
        num: i32,
        py_args: &Bound<'_, PyTuple>,
        name: &str,
        py_kwargs: Option<&Bound<'_, PyDict>>,
    ) -> String {
        let num_before = self.num;
        format!(
            "num_before: {}, num: {}, args: {:?}, name: {}, kwargs: {:?}",
            num_before, num, py_args, name, py_kwargs
        )
    }

    fn make_change(&mut self, num: i32) -> PyResult<()> {
        self.num = num;
        Ok(format!("num={}", self.num))
    }
}

#[pyfunction]
#[pyo3(signature = (lambda))]
pub fn simple_python_bound_function(py: Python<'_>, lambda: PyObject) -> PyResult<()> {
    Ok(())
}

fn get_length(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    obj.len()
}

#[pyfunction]
fn object_length(#[pyo3(from_py_with = "get_length")] argment: usize) -> usize {
    argment
}
