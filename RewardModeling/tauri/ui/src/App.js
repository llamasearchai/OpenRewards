import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import { open } from '@tauri-apps/api/dialog';
import { 
  Routes, 
  Route, 
  BrowserRouter, 
  useNavigate, 
  Navigate 
} from 'react-router-dom';
import Dashboard from './views/Dashboard';
import ExperimentDetail from './views/ExperimentDetail';
import NewExperiment from './views/NewExperiment';
import DatasetViewer from './views/DatasetViewer';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import './styles/App.css';

function AppContent() {
  const [experiments, setExperiments] = useState([]);
  const [currentExperiment, setCurrentExperiment] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    loadExperiments();
  }, []);

  const loadExperiments = async () => {
    try {
      setLoading(true);
      const experiments = await invoke('get_experiments');
      setExperiments(experiments);
      
      const current = await invoke('get_current_experiment');
      setCurrentExperiment(current);
      
      setLoading(false);
    } catch (error) {
      console.error('Failed to load experiments:', error);
      setLoading(false);
    }
  };

  const handleCreateExperiment = async (experimentData) => {
    try {
      const newExperiment = await invoke('create_experiment', {
        name: experimentData.name,
        description: experimentData.description,
        modelName: experimentData.modelName,
        datasetPath: experimentData.datasetPath,
      });
      
      await invoke('set_current_experiment', { id: newExperiment.id });
      
      await loadExperiments();
      navigate(`/experiment/${newExperiment.id}`);
      
      return newExperiment;
    } catch (error) {
      console.error('Failed to create experiment:', error);
      throw error;
    }
  };

  const handleStartExperiment = async (experimentId) => {
    try {
      await invoke('start_experiment', { id: experimentId });
      await loadExperiments();
    } catch (error) {
      console.error('Failed to start experiment:', error);
      throw error;
    }
  };

  const handleSelectExperiment = async (experimentId) => {
    try {
      await invoke('set_current_experiment', { id: experimentId });
      await loadExperiments();
      navigate(`/experiment/${experimentId}`);
    } catch (error) {
      console.error('Failed to select experiment:', error);
    }
  };

  const handleOpenDatasetDialog = async () => {
    try {
      const selected = await open({
        directory: false,
        multiple: false,
        filters: [{
          name: 'Dataset',
          extensions: ['jsonl', 'json', 'csv']
        }]
      });
      
      if (selected) {
        return selected;
      }
    } catch (error) {
      console.error('Failed to open dialog:', error);
    }
    return null;
  };

  if (loading) {
    return <div className="loading">Loading application...</div>;
  }

  return (
    <div className="app-container">
      <Header />
      <div className="content-container">
        <Sidebar 
          experiments={experiments} 
          currentExperiment={currentExperiment}
          onSelectExperiment={handleSelectExperiment}
          onNewExperiment={() => navigate('/new-experiment')}
        />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard experiments={experiments} />} />
            <Route 
              path="/experiment/:id" 
              element={
                <ExperimentDetail 
                  onStartExperiment={handleStartExperiment}
                  onRefresh={loadExperiments}
                />
              } 
            />
            <Route 
              path="/new-experiment" 
              element={
                <NewExperiment 
                  onCreateExperiment={handleCreateExperiment}
                  onOpenDatasetDialog={handleOpenDatasetDialog}
                />
              } 
            />
            <Route path="/dataset-viewer" element={<DatasetViewer />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  );
}

export default App;