use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use crate::data_processing::{parser::PreferencePair, parser::PreferenceDataParser, parser::TokenizationConfig};

/// Python module for the Rust data processing components
#[pymodule]
fn reward_modeling_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyPreferencePair>()?;
    m.add_class::<PyPreferenceDataParser>()?;
    m.add_class::<PyTokenizationConfig>()?;
    Ok(())
}

/// Python wrapper for PreferencePair
#[pyclass]
#[derive(Clone)]
struct PyPreferencePair {
    #[pyo3(get, set)]
    id: String,
    
    #[pyo3(get, set)]
    prompt: String,
    
    #[pyo3(get, set)]
    chosen: String,
    
    #[pyo3(get, set)]
    rejected: String,
    
    #[pyo3(get, set)]
    metadata: Option<HashMap<String, String>>,
}

#[pymethods]
impl PyPreferencePair {
    #[new]
    fn new(id: String, prompt: String, chosen: String, rejected: String, metadata: Option<HashMap<String, String>>) -> Self {
        Self {
            id,
            prompt,
            chosen,
            rejected,
            metadata,
        }
    }
    
    /// Convert to a Python dictionary
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("id", self.id.clone())?;
        dict.set_item("prompt", self.prompt.clone())?;
        dict.set_item("chosen", self.chosen.clone())?;
        dict.set_item("rejected", self.rejected.clone())?;
        
        if let Some(metadata) = &self.metadata {
            let meta_dict = PyDict::new(py);
            for (k, v) in metadata {
                meta_dict.set_item(k, v)?;
            }
            dict.set_item("metadata", meta_dict)?;
        } else {
            dict.set_item("metadata", py.None())?;
        }
        
        Ok(dict.into())
    }
    
    /// Create from a Python dictionary
    #[staticmethod]
    fn from_dict(dict: &PyDict) -> PyResult<Self> {
        let id = dict.get_item("id")
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'id' field"))?
            .extract::<String>()?;
            
        let prompt = dict.get_item("prompt")
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'prompt' field"))?
            .extract::<String>()?;
            
        let chosen = dict.get_item("chosen")
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'chosen' field"))?
            .extract::<String>()?;
            
        let rejected = dict.get_item("rejected")
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'rejected' field"))?
            .extract::<String>()?;
            
        let metadata = if let Some(meta_dict) = dict.get_item("metadata") {
            if meta_dict.is_none() {
                None
            } else {
                let py_dict = meta_dict.downcast::<PyDict>()?;
                let mut metadata = HashMap::new();
                for (k, v) in py_dict.iter() {
                    metadata.insert(k.extract::<String>()?, v.extract::<String>()?);
                }
                Some(metadata)
            }
        } else {
            None
        };
        
        Ok(Self {
            id,
            prompt,
            chosen,
            rejected,
            metadata,
        })
    }
}

/// Convert between Rust PreferencePair and Python PyPreferencePair
impl From<PreferencePair> for PyPreferencePair {
    fn from(pair: PreferencePair) -> Self {
        Self {
            id: pair.id,
            prompt: pair.prompt,
            chosen: pair.chosen,
            rejected: pair.rejected,
            metadata: pair.metadata,
        }
    }
}

impl From<PyPreferencePair> for PreferencePair {
    fn from(pair: PyPreferencePair) -> Self {
        Self {
            id: pair.id,
            prompt: pair.prompt,
            chosen: pair.chosen,
            rejected: pair.rejected,
            metadata: pair.metadata,
        }
    }
}

/// Python wrapper for TokenizationConfig
#[pyclass]
#[derive(Clone)]
struct PyTokenizationConfig {
    #[pyo3(get, set)]
    max_seq_length: usize,
    
    #[pyo3(get, set)]
    max_prompt_length: usize,
    
    #[pyo3(get, set)]
    truncate: bool,
    
    #[pyo3(get, set)]
    filter_tokens: Option<Vec<String>>,
}

#[pymethods]
impl PyTokenizationConfig {
    #[new]
    fn new(max_seq_length: usize, max_prompt_length: usize, truncate: bool, filter_tokens: Option<Vec<String>>) -> Self {
        Self {
            max_seq_length,
            max_prompt_length,
            truncate,
            filter_tokens,
        }
    }
    
    /// Create with default values
    #[staticmethod]
    fn default() -> Self {
        Self {
            max_seq_length: 2048,
            max_prompt_length: 512,
            truncate: true,
            filter_tokens: None,
        }
    }
}

impl From<PyTokenizationConfig> for TokenizationConfig {
    fn from(config: PyTokenizationConfig) -> Self {
        Self {
            max_seq_length: config.max_seq_length,
            max_prompt_length: config.max_prompt_length,
            truncate: config.truncate,
            filter_tokens: config.filter_tokens,
        }
    }
}

/// Python wrapper for PreferenceDataParser
#[pyclass]
struct PyPreferenceDataParser {
    parser: Arc<PreferenceDataParser>,
}

#[pymethods]
impl PyPreferenceDataParser {
    #[new]
    fn new(config: Option<PyTokenizationConfig>) -> Self {
        let rust_config = config.map(TokenizationConfig::from);
        Self {
            parser: Arc::new(PreferenceDataParser::new(rust_config)),
        }
    }
    
    /// Parse a JSONL file containing preference pairs
    fn parse_jsonl(&self, path: String) -> PyResult<Vec<PyPreferencePair>> {
        match self.parser.parse_jsonl(path) {
            Ok(pairs) => Ok(pairs.into_iter().map(PyPreferencePair::from).collect()),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error parsing JSONL: {}", e))),
        }
    }
    
    /// Process a vector of preference pairs
    fn process_pairs(&self, py_pairs: Vec<PyPreferencePair>) -> Vec<PyPreferencePair> {
        let rust_pairs: Vec<PreferencePair> = py_pairs.into_iter().map(PreferencePair::from).collect();
        self.parser.process_pairs(&rust_pairs)
            .into_iter()
            .map(PyPreferencePair::from)
            .collect()
    }
    
    /// Save processed pairs to a JSONL file
    fn save_jsonl(&self, py_pairs: Vec<PyPreferencePair>, path: String) -> PyResult<()> {
        let rust_pairs: Vec<PreferencePair> = py_pairs.into_iter().map(PreferencePair::from).collect();
        match self.parser.save_jsonl(&rust_pairs, path) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error saving JSONL: {}", e))),
        }
    }
}