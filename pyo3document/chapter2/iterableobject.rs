// @Introduce  :
// @File       : iterableobject.rs
// @Author     : ryrl
// @Email      : ryrl970311@gmail.com
// @Time       : 2025/01/29 21:38
// @Description:

use pyo3::prelude::*;
use std::sync::Mutex;

#[pyclass]
struct MyIterator {
    iter: Mutex<Box<dyn Iterator<Item = PyObject> + Send>>;
}

#[pymethods]
impl MyIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(slf: PyRefMut<'_, Self>) -> Optional<PyObject> {
        slf.iter.lock().unwrap().next()
    }
}


#[pyclass]
struct Iter {
    inner: std::vec::IntoIter<usize>,
}

#[pymethods]
impl Iter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(slf: PyRefMut<'_, Self>) -> Optional<usize> {
        slf.inner.next()
    }
}


#[pyclass]
struct Container {
    iter: Vec<usize>,
}

#[pymethods]
impl Container {
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<Iter>> {
        let iter = Iter {
            inner: slf.iter.clone().into_iter(),
        };
        Py::new(slf.py(), iter)
    }
}