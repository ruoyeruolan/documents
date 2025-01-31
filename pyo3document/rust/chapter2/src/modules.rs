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

#[pymodule]
fn parent_module(m: &Bound<'_, PyModule>) -> Pyresult<()> {
    register_child_module(m)?;
    Ok(())
}

fn register_child_module(parent: &Bound<'_, PyModule>) -> Pyresult<()> {
    let child_module = PyModule::new(parent.py(), "child_module")?;
    child_module.add_function(wrap_pyfunction!(func, &child_module)?)?;
    parent.add_submodule(&child_module)?;
}

#[pyfunction]
fn func() -> String {
    "func".to_string()
}

#[pymodule]
mod my_extension {
    use super::*;

    #[pymodule_export]
    use super::double;

    #[pyfunction]
    fn triple(s: usize) -> usize {
        x * 3
    }

    #[pyclass]
    struct Unit;

    #[pymodule]
    mod submodule {
        unimplemented!();
    }

    #[pymodule_init]
    fn init(m: Bound<'_, PyModule>) -> Pyresult<()> {
        m.add("double2", m.getattr("double")?)
    }
}
