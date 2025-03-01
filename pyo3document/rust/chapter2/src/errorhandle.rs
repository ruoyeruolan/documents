// @Introduce  :
// @File       : errorhandle.rs
// @Author     : ryrl
// @Email      : ryrl970311@gmail.com
// @Time       : 2025/01/31 15:18
// @Description:

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use some_crate::{get_x, OtherError};
use std::fmt;
use std::num::ParseIntError;

#[pyfunction]
fn check_positive(x: i32) -> PyResult<()> {
    if x <= 0 {
        Err(PyValueError::new_err("Value must be positive"))
    } else {
        Ok(())
    }
}

#[pyfunction]
fn parse_int(x: &str) -> Result<usize, ParseIntError> {
    x.parse()
}

#[derive(Debug)]
struct CustomIOError;

impl std::error::Error for CustomIOError {}

impl fmt::Display for CustomIOError {
    fn fmt(&self, f: &mut fnt::Formatter) -> fmt::Result {
        write!(f, "Custom IO Error")
    }
}

impl std::convert::From<CustomIOError> for PyErr {
    fn from(err: CustomIOError) -> PyErr {
        PyOSError::new_err(err.to_string())
    }
}

pub struct Connection {/* ... */}

fn bind(addr: String) -> Result<Connection, CustomIOError> {
    if &addr == "0.0.0.0" {
        Err(CustomIOError)
    } else {
        Ok(Connection {/* ... */})
    }
}

#[pyfunction]
fn connect(s: String) -> Resulr<(), CustomIOError> {
    bind(s)?;
    Ok(())
}

struct MyotherError(OtherError);

impl From<MyotherError> for PyErr {
    fn from(error: MyotherError) -> Self {
        PyValueError::new_err(error.0.message())
    }
}

impl From<OtherError> for MyOtherError {
    fn from(other: OtherError) -> Self {
        Self(other)
    }
}

#[pyfunction]
fn wrapped_get_x() -> Result<i32, MyotherError> {
    let x: i32 = get_x()?;
    Ok(x)
}
