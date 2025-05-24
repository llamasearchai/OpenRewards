use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;

/// Represents a preference pair for reward modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferencePair {
    /// Unique identifier for this preference pair
    pub id: String,
    
    /// Text prompt/input
    pub prompt: String,
    
    /// Preferred/chosen completion
    pub chosen: String,
    
    /// Rejected completion
    pub rejected: String,
    
    /// Optional metadata about this sample
    pub metadata: Option<HashMap<String, String>>,
}

/// Handles parsing and processing of preference datasets
pub struct PreferenceDataParser {
    /// Configuration for tokenization and filtering
    tokenization_config: TokenizationConfig,
    
    /// Cache for processed data to avoid redundant work
    cache: Arc<Mutex<HashMap<String, PreferencePair>>>,
}

/// Configuration for text tokenization and processing
#[derive(Debug, Clone)]
pub struct TokenizationConfig {
    /// Maximum sequence length for combined prompt + completion
    pub max_seq_length: usize,
    
    /// Maximum prompt length
    pub max_prompt_length: usize,
    
    /// Whether to truncate sequences that exceed max length
    pub truncate: bool,
    
    /// Optional set of tokens to filter out
    pub filter_tokens: Option<Vec<String>>,
}

impl Default for TokenizationConfig {
    fn default() -> Self {
        Self {
            max_seq_length: 2048,
            max_prompt_length: 512,
            truncate: true,
            filter_tokens: None,
        }
    }
}

impl PreferenceDataParser {
    /// Create a new parser with the specified configuration
    pub fn new(config: Option<TokenizationConfig>) -> Self {
        Self {
            tokenization_config: config.unwrap_or_default(),
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    /// Parse a JSONL file containing preference pairs
    pub fn parse_jsonl<P: AsRef<Path>>(&self, path: P) -> Result<Vec<PreferencePair>, Box<dyn Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        
        let pairs: Result<Vec<PreferencePair>, _> = reader
            .lines()
            .map(|line| -> Result<PreferencePair, Box<dyn Error>> {
                let line = line?;
                let pair: PreferencePair = serde_json::from_str(&line)?;
                Ok(pair)
            })
            .collect();
            
        pairs
    }
    
    /// Process a vector of preference pairs in parallel
    pub fn process_pairs(&self, pairs: &[PreferencePair]) -> Vec<PreferencePair> {
        pairs.par_iter()
            .filter_map(|pair| self.process_pair(pair).ok())
            .collect()
    }
    
    /// Process a single preference pair
    pub fn process_pair(&self, pair: &PreferencePair) -> Result<PreferencePair, Box<dyn Error>> {
        // Check cache first
        {
            let cache = self.cache.lock().unwrap();
            if let Some(cached_pair) = cache.get(&pair.id) {
                return Ok(cached_pair.clone());
            }
        }
        
        // Apply filtering and truncation based on config
        let mut processed = pair.clone();
        
        // Truncate prompt if needed
        if self.tokenization_config.truncate && self.simple_tokenize(&processed.prompt).len() > self.tokenization_config.max_prompt_length {
            processed.prompt = self.truncate_text(&processed.prompt, self.tokenization_config.max_prompt_length);
        }
        
        // Truncate completions if needed
        let max_completion_length = self.tokenization_config.max_seq_length - self.simple_tokenize(&processed.prompt).len();
        
        if self.tokenization_config.truncate && self.simple_tokenize(&processed.chosen).len() > max_completion_length {
            processed.chosen = self.truncate_text(&processed.chosen, max_completion_length);
        }
        
        if self.tokenization_config.truncate && self.simple_tokenize(&processed.rejected).len() > max_completion_length {
            processed.rejected = self.truncate_text(&processed.rejected, max_completion_length);
        }
        
        // Apply token filtering if configured
        if let Some(filter_tokens) = &self.tokenization_config.filter_tokens {
            for token in filter_tokens {
                processed.prompt = processed.prompt.replace(token, "");
                processed.chosen = processed.chosen.replace(token, "");
                processed.rejected = processed.rejected.replace(token, "");
            }
        }
        
        // Cache processed result
        {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(processed.id.clone(), processed.clone());
        }
        
        Ok(processed)
    }
    
    /// Save processed pairs to a JSONL file
    pub fn save_jsonl<P: AsRef<Path>>(&self, pairs: &[PreferencePair], path: P) -> Result<(), Box<dyn Error>> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        
        for pair in pairs {
            let json = serde_json::to_string(pair)?;
            writeln!(writer, "{}", json)?;
        }
        
        writer.flush()?;
        Ok(())
    }
    
    /// Simple whitespace tokenization for length estimation
    fn simple_tokenize(&self, text: &str) -> Vec<&str> {
        text.split_whitespace().collect()
    }
    
    /// Truncate text to the specified number of tokens
    fn truncate_text(&self, text: &str, max_tokens: usize) -> String {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        if tokens.len() <= max_tokens {
            return text.to_string();
        }
        
        tokens[..max_tokens].join(" ")
    }
}