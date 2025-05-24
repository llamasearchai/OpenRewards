#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex};
use tauri::{Manager, State};
use tauri_plugin_log::{LogTarget, LoggerBuilder};

// AppState to maintain application state across invocations
#[derive(Default)]
struct AppState {
    experiments: Arc<Mutex<Vec<Experiment>>>,
    current_experiment: Arc<Mutex<Option<String>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Experiment {
    id: String,
    name: String,
    description: String,
    model_name: String,
    dataset_path: String,
    created_at: String,
    status: String,
    metrics: Option<ExperimentMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExperimentMetrics {
    loss: f64,
    accuracy: f64,
    reward_gap: f64,
    eval_results: Option<serde_json::Value>,
}

#[tauri::command]
fn get_experiments(state: State<AppState>) -> Result<Vec<Experiment>, String> {
    let experiments = state.experiments.lock().unwrap();
    Ok(experiments.clone())
}

#[tauri::command]
fn get_current_experiment(state: State<AppState>) -> Result<Option<Experiment>, String> {
    let current_id = state.current_experiment.lock().unwrap();
    if let Some(id) = &*current_id {
        let experiments = state.experiments.lock().unwrap();
        let experiment = experiments.iter().find(|e| e.id == *id).cloned();
        Ok(experiment)
    } else {
        Ok(None)
    }
}

#[tauri::command]
fn create_experiment(
    state: State<AppState>,
    name: String,
    description: String,
    model_name: String,
    dataset_path: String,
) -> Result<Experiment, String> {
    // Validate inputs
    if name.is_empty() {
        return Err("Experiment name cannot be empty".to_string());
    }
    
    if !Path::new(&dataset_path).exists() {
        return Err(format!("Dataset path '{}' does not exist", dataset_path));
    }
    
    // Create new experiment
    let now = chrono::Local::now().to_rfc3339();
    let experiment = Experiment {
        id: uuid::Uuid::new_v4().to_string(),
        name,
        description,
        model_name,
        dataset_path,
        created_at: now,
        status: "created".to_string(),
        metrics: None,
    };
    
    // Update state
    {
        let mut experiments = state.experiments.lock().unwrap();
        experiments.push(experiment.clone());
    }
    
    Ok(experiment)
}

#[tauri::command]
fn set_current_experiment(state: State<AppState>, id: String) -> Result<(), String> {
    let experiments = state.experiments.lock().unwrap();
    if !experiments.iter().any(|e| e.id == id) {
        return Err(format!("Experiment with id '{}' not found", id));
    }
    
    let mut current_id = state.current_experiment.lock().unwrap();
    *current_id = Some(id);
    
    Ok(())
}

#[tauri::command]
fn start_experiment(state: State<AppState>, id: String) -> Result<(), String> {
    // Find experiment
    let experiment = {
        let mut experiments = state.experiments.lock().unwrap();
        let experiment = experiments.iter_mut().find(|e| e.id == id);
        
        if let Some(experiment) = experiment {
            experiment.status = "running".to_string();
            experiment.clone()
        } else {
            return Err(format!("Experiment with id '{}' not found", id));
        }
    };
    
    // Launch Python process to run the experiment
    // This is a simplified example - in production you'd use a more robust approach
    std::thread::spawn(move || {
        let output = Command::new("python")
            .arg("-m")
            .arg("reward_modeling.training.run_experiment")
            .arg("--experiment-id")
            .arg(&experiment.id)
            .arg("--model-name")
            .arg(&experiment.model_name)
            .arg("--dataset-path")
            .arg(&experiment.dataset_path)
            .output();
            
        match output {
            Ok(_) => {
                // Update experiment status
                // In a real implementation, you'd parse the output and update metrics
            }
            Err(e) => {
                eprintln!("Failed to start experiment: {}", e);
            }
        }
    });
    
    Ok(())
}

#[tauri::command]
fn get_dataset_preview(dataset_path: String, limit: usize) -> Result<Vec<serde_json::Value>, String> {
    if !Path::new(&dataset_path).exists() {
        return Err(format!("Dataset path '{}' does not exist", dataset_path));
    }
    
    // Read dataset file
    let content = fs::read_to_string(&dataset_path)
        .map_err(|e| format!("Failed to read dataset: {}", e))?;
        
    // Parse JSONL format
    let mut samples = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i >= limit {
            break;
        }
        
        let value: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| format!("Failed to parse line {}: {}", i + 1, e))?;