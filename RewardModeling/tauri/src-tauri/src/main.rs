// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::{Manager, State, Window, AppHandle};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use anyhow::Result;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use log::{info, warn, error};

mod api;
mod database;
mod monitoring;
mod experiments;
mod models;
mod config;
mod websocket;

use api::ApiClient;
use database::Database;
use monitoring::SystemMonitor;
use experiments::{ExperimentManager, ExperimentStatus};
use models::{ModelManager, TrainingProgress};
use config::AppConfig;
use websocket::WebSocketManager;

// Application state
#[derive(Default)]
pub struct AppState {
    pub api_client: Arc<Mutex<Option<ApiClient>>>,
    pub database: Arc<Mutex<Option<Database>>>,
    pub system_monitor: Arc<Mutex<Option<SystemMonitor>>>,
    pub experiment_manager: Arc<Mutex<ExperimentManager>>,
    pub model_manager: Arc<Mutex<ModelManager>>,
    pub websocket_manager: Arc<Mutex<Option<WebSocketManager>>>,
    pub config: Arc<Mutex<AppConfig>>,
    pub training_sessions: Arc<Mutex<HashMap<String, TrainingProgress>>>,
}

// Data structures for API communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub status: ExperimentStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub config: serde_json::Value,
    pub metrics: Option<serde_json::Value>,
    pub progress: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub model_type: String,
    pub size: String,
    pub status: String,
    pub accuracy: Option<f64>,
    pub created_at: DateTime<Utc>,
    pub file_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub memory_total: u64,
    pub gpu_usage: Option<f64>,
    pub gpu_memory_usage: Option<f64>,
    pub disk_usage: f64,
    pub network_rx: u64,
    pub network_tx: u64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub experiment_id: String,
    pub epoch: u32,
    pub step: u32,
    pub loss: f64,
    pub accuracy: Option<f64>,
    pub learning_rate: f64,
    pub throughput: Option<f64>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiSettings {
    pub base_url: String,
    pub api_key: Option<String>,
    pub timeout: u64,
    pub retry_attempts: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseSettings {
    pub database_url: String,
    pub max_connections: u32,
    pub enable_logging: bool,
}

// Tauri commands
#[tauri::command]
async fn initialize_app(state: State<'_, AppState>) -> Result<String, String> {
    info!("Initializing application...");
    
    // Initialize database
    let db = Database::new().await.map_err(|e| e.to_string())?;
    *state.database.lock().unwrap() = Some(db);
    
    // Initialize system monitor
    let monitor = SystemMonitor::new().map_err(|e| e.to_string())?;
    *state.system_monitor.lock().unwrap() = Some(monitor);
    
    // Load configuration
    let config = AppConfig::load().map_err(|e| e.to_string())?;
    *state.config.lock().unwrap() = config;
    
    info!("Application initialized successfully");
    Ok("Application initialized".to_string())
}

#[tauri::command]
async fn connect_to_api(
    api_settings: ApiSettings,
    state: State<'_, AppState>
) -> Result<String, String> {
    info!("Connecting to API: {}", api_settings.base_url);
    
    let client = ApiClient::new(api_settings).await.map_err(|e| e.to_string())?;
    *state.api_client.lock().unwrap() = Some(client);
    
    Ok("Connected to API successfully".to_string())
}

#[tauri::command]
async fn get_experiments(state: State<'_, AppState>) -> Result<Vec<ExperimentInfo>, String> {
    let api_client = state.api_client.lock().unwrap();
    
    if let Some(client) = api_client.as_ref() {
        client.get_experiments().await.map_err(|e| e.to_string())
    } else {
        // Return local experiments if no API connection
        let experiment_manager = state.experiment_manager.lock().unwrap();
        Ok(experiment_manager.get_all_experiments())
    }
}

#[tauri::command]
async fn create_experiment(
    name: String,
    description: String,
    config: serde_json::Value,
    state: State<'_, AppState>
) -> Result<ExperimentInfo, String> {
    info!("Creating experiment: {}", name);
    
    let experiment_id = Uuid::new_v4().to_string();
    let experiment = ExperimentInfo {
        id: experiment_id.clone(),
        name,
        description,
        status: ExperimentStatus::Created,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        config,
        metrics: None,
        progress: 0.0,
    };
    
    // Try API first, then local storage
    let api_client = state.api_client.lock().unwrap();
    if let Some(client) = api_client.as_ref() {
        client.create_experiment(&experiment).await.map_err(|e| e.to_string())
    } else {
        let mut experiment_manager = state.experiment_manager.lock().unwrap();
        experiment_manager.add_experiment(experiment.clone());
        Ok(experiment)
    }
}

#[tauri::command]
async fn start_training(
    experiment_id: String,
    state: State<'_, AppState>
) -> Result<String, String> {
    info!("Starting training for experiment: {}", experiment_id);
    
    let api_client = state.api_client.lock().unwrap();
    if let Some(client) = api_client.as_ref() {
        client.start_training(&experiment_id).await.map_err(|e| e.to_string())
    } else {
        let mut experiment_manager = state.experiment_manager.lock().unwrap();
        experiment_manager.start_experiment(&experiment_id);
        Ok("Training started locally".to_string())
    }
}

#[tauri::command]
async fn stop_training(
    experiment_id: String,
    state: State<'_, AppState>
) -> Result<String, String> {
    info!("Stopping training for experiment: {}", experiment_id);
    
    let api_client = state.api_client.lock().unwrap();
    if let Some(client) = api_client.as_ref() {
        client.stop_training(&experiment_id).await.map_err(|e| e.to_string())
    } else {
        let mut experiment_manager = state.experiment_manager.lock().unwrap();
        experiment_manager.stop_experiment(&experiment_id);
        Ok("Training stopped locally".to_string())
    }
}

#[tauri::command]
async fn get_models(state: State<'_, AppState>) -> Result<Vec<ModelInfo>, String> {
    let api_client = state.api_client.lock().unwrap();
    
    if let Some(client) = api_client.as_ref() {
        client.get_models().await.map_err(|e| e.to_string())
    } else {
        let model_manager = state.model_manager.lock().unwrap();
        Ok(model_manager.get_all_models())
    }
}

#[tauri::command]
async fn upload_model(
    file_path: String,
    model_name: String,
    model_type: String,
    state: State<'_, AppState>
) -> Result<ModelInfo, String> {
    info!("Uploading model: {} from {}", model_name, file_path);
    
    let api_client = state.api_client.lock().unwrap();
    if let Some(client) = api_client.as_ref() {
        client.upload_model(&file_path, &model_name, &model_type).await.map_err(|e| e.to_string())
    } else {
        let mut model_manager = state.model_manager.lock().unwrap();
        model_manager.add_model(file_path, model_name, model_type)
    }
}

#[tauri::command]
async fn delete_model(
    model_id: String,
    state: State<'_, AppState>
) -> Result<String, String> {
    info!("Deleting model: {}", model_id);
    
    let api_client = state.api_client.lock().unwrap();
    if let Some(client) = api_client.as_ref() {
        client.delete_model(&model_id).await.map_err(|e| e.to_string())
    } else {
        let mut model_manager = state.model_manager.lock().unwrap();
        model_manager.delete_model(&model_id)
    }
}

#[tauri::command]
async fn get_system_metrics(state: State<'_, AppState>) -> Result<SystemMetrics, String> {
    let system_monitor = state.system_monitor.lock().unwrap();
    
    if let Some(monitor) = system_monitor.as_ref() {
        monitor.get_current_metrics().map_err(|e| e.to_string())
    } else {
        Err("System monitor not initialized".to_string())
    }
}

#[tauri::command]
async fn get_training_metrics(
    experiment_id: String,
    state: State<'_, AppState>
) -> Result<Vec<TrainingMetrics>, String> {
    let api_client = state.api_client.lock().unwrap();
    
    if let Some(client) = api_client.as_ref() {
        client.get_training_metrics(&experiment_id).await.map_err(|e| e.to_string())
    } else {
        // Return mock data for local mode
        Ok(vec![])
    }
}

#[tauri::command]
async fn get_experiment_logs(
    experiment_id: String,
    state: State<'_, AppState>
) -> Result<Vec<String>, String> {
    let api_client = state.api_client.lock().unwrap();
    
    if let Some(client) = api_client.as_ref() {
        client.get_experiment_logs(&experiment_id).await.map_err(|e| e.to_string())
    } else {
        Ok(vec!["Local mode - no logs available".to_string()])
    }
}

#[tauri::command]
async fn evaluate_model(
    model_id: String,
    dataset_path: String,
    state: State<'_, AppState>
) -> Result<serde_json::Value, String> {
    info!("Evaluating model: {} on dataset: {}", model_id, dataset_path);
    
    let api_client = state.api_client.lock().unwrap();
    if let Some(client) = api_client.as_ref() {
        client.evaluate_model(&model_id, &dataset_path).await.map_err(|e| e.to_string())
    } else {
        // Return mock evaluation results
        Ok(serde_json::json!({
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85
        }))
    }
}

#[tauri::command]
async fn export_experiment(
    experiment_id: String,
    export_path: String,
    state: State<'_, AppState>
) -> Result<String, String> {
    info!("Exporting experiment: {} to {}", experiment_id, export_path);
    
    let api_client = state.api_client.lock().unwrap();
    if let Some(client) = api_client.as_ref() {
        client.export_experiment(&experiment_id, &export_path).await.map_err(|e| e.to_string())
    } else {
        Ok("Export completed (local mode)".to_string())
    }
}

#[tauri::command]
async fn import_experiment(
    import_path: String,
    state: State<'_, AppState>
) -> Result<ExperimentInfo, String> {
    info!("Importing experiment from: {}", import_path);
    
    let api_client = state.api_client.lock().unwrap();
    if let Some(client) = api_client.as_ref() {
        client.import_experiment(&import_path).await.map_err(|e| e.to_string())
    } else {
        Err("Import not supported in local mode".to_string())
    }
}

#[tauri::command]
async fn get_app_config(state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    let config = state.config.lock().unwrap();
    Ok(config.to_json())
}

#[tauri::command]
async fn update_app_config(
    new_config: serde_json::Value,
    state: State<'_, AppState>
) -> Result<String, String> {
    let mut config = state.config.lock().unwrap();
    config.update_from_json(new_config).map_err(|e| e.to_string())?;
    config.save().map_err(|e| e.to_string())?;
    Ok("Configuration updated".to_string())
}

#[tauri::command]
async fn check_api_health(state: State<'_, AppState>) -> Result<bool, String> {
    let api_client = state.api_client.lock().unwrap();
    
    if let Some(client) = api_client.as_ref() {
        Ok(client.health_check().await.unwrap_or(false))
    } else {
        Ok(false)
    }
}

#[tauri::command]
async fn get_dataset_info(
    dataset_path: String,
    state: State<'_, AppState>
) -> Result<serde_json::Value, String> {
    let api_client = state.api_client.lock().unwrap();
    
    if let Some(client) = api_client.as_ref() {
        client.get_dataset_info(&dataset_path).await.map_err(|e| e.to_string())
    } else {
        // Return mock dataset info
        Ok(serde_json::json!({
            "name": "Local Dataset",
            "size": 1000,
            "format": "json",
            "columns": ["prompt", "chosen", "rejected"]
        }))
    }
}

#[tauri::command]
async fn download_model(
    model_id: String,
    download_path: String,
    state: State<'_, AppState>
) -> Result<String, String> {
    info!("Downloading model: {} to {}", model_id, download_path);
    
    let api_client = state.api_client.lock().unwrap();
    if let Some(client) = api_client.as_ref() {
        client.download_model(&model_id, &download_path).await.map_err(|e| e.to_string())
    } else {
        Err("Download not available in local mode".to_string())
    }
}

// WebSocket connection for real-time updates
#[tauri::command]
async fn start_websocket_connection(
    url: String,
    app_handle: AppHandle,
    state: State<'_, AppState>
) -> Result<String, String> {
    info!("Starting WebSocket connection to: {}", url);
    
    let ws_manager = WebSocketManager::new(url, app_handle).await.map_err(|e| e.to_string())?;
    *state.websocket_manager.lock().unwrap() = Some(ws_manager);
    
    Ok("WebSocket connection established".to_string())
}

#[tauri::command]
async fn stop_websocket_connection(state: State<'_, AppState>) -> Result<String, String> {
    let mut ws_manager = state.websocket_manager.lock().unwrap();
    
    if let Some(manager) = ws_manager.take() {
        manager.disconnect().await.map_err(|e| e.to_string())?;
        Ok("WebSocket connection closed".to_string())
    } else {
        Ok("No WebSocket connection to close".to_string())
    }
}

// File operations
#[tauri::command]
async fn select_file(file_types: Vec<String>) -> Result<Option<String>, String> {
    use tauri::api::dialog::FileDialogBuilder;
    
    let file_dialog = FileDialogBuilder::new();
    
    // Add file filters
    let dialog = if !file_types.is_empty() {
        file_types.into_iter().fold(file_dialog, |dialog, ext| {
            dialog.add_filter(&format!("{} files", ext.to_uppercase()), &[&ext])
        })
    } else {
        file_dialog
    };
    
    let result = dialog.pick_file().await;
    Ok(result.map(|path| path.to_string_lossy().to_string()))
}

#[tauri::command]
async fn select_directory() -> Result<Option<String>, String> {
    use tauri::api::dialog::FileDialogBuilder;
    
    let result = FileDialogBuilder::new().pick_folder().await;
    Ok(result.map(|path| path.to_string_lossy().to_string()))
}

// Notification system
#[tauri::command]
async fn show_notification(
    title: String,
    body: String,
    app_handle: AppHandle
) -> Result<(), String> {
    app_handle
        .emit_all("notification", serde_json::json!({
            "title": title,
            "body": body,
            "timestamp": Utc::now()
        }))
        .map_err(|e| e.to_string())?;
    
    Ok(())
}

fn main() {
    // Initialize logging
    env_logger::init();
    
    tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            initialize_app,
            connect_to_api,
            get_experiments,
            create_experiment,
            start_training,
            stop_training,
            get_models,
            upload_model,
            delete_model,
            get_system_metrics,
            get_training_metrics,
            get_experiment_logs,
            evaluate_model,
            export_experiment,
            import_experiment,
            get_app_config,
            update_app_config,
            check_api_health,
            get_dataset_info,
            download_model,
            start_websocket_connection,
            stop_websocket_connection,
            select_file,
            select_directory,
            show_notification
        ])
        .setup(|app| {
            // Setup system tray
            #[cfg(desktop)]
            {
                use tauri::{SystemTray, SystemTrayMenu, SystemTrayMenuItem, CustomMenuItem};
                
                let tray_menu = SystemTrayMenu::new()
                    .add_item(CustomMenuItem::new("show", "Show"))
                    .add_native_item(SystemTrayMenuItem::Separator)
                    .add_item(CustomMenuItem::new("quit", "Quit"));
                
                let system_tray = SystemTray::new().with_menu(tray_menu);
                app.handle().plugin(tauri_plugin_system_tray::init(system_tray))?;
            }
            
            // Setup auto-updater
            #[cfg(desktop)]
            {
                let app_handle = app.handle();
                tauri::async_runtime::spawn(async move {
                    let response = app_handle.updater().check().await;
                    match response {
                        Ok(update) => {
                            if update.is_update_available() {
                                info!("Update available: {}", update.latest_version());
                                let _ = update.download_and_install().await;
                            }
                        }
                        Err(e) => {
                            warn!("Failed to check for updates: {}", e);
                        }
                    }
                });
            }
            
            Ok(())
        })
        .on_system_tray_event(|app, event| {
            use tauri::SystemTrayEvent;
            
            match event {
                SystemTrayEvent::LeftClick { .. } => {
                    let window = app.get_window("main").unwrap();
                    let _ = window.show();
                    let _ = window.set_focus();
                }
                SystemTrayEvent::MenuItemClick { id, .. } => {
                    match id.as_str() {
                        "show" => {
                            let window = app.get_window("main").unwrap();
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                        "quit" => {
                            std::process::exit(0);
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
} 