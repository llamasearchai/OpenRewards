import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import { QueryClient, QueryClientProvider } from 'react-query';
import { ReactQueryDevtools } from 'react-query/devtools';
import { SnackbarProvider } from 'notistack';
import { ErrorBoundary } from 'react-error-boundary';
import { invoke } from '@tauri-apps/api/tauri';

// Components
import { AppLayout } from './components/layout/AppLayout';
import { LoadingScreen } from './components/common/LoadingScreen';
import { ErrorFallback } from './components/common/ErrorFallback';
import { NotificationManager } from './components/common/NotificationManager';

// Pages
import { Dashboard } from './pages/Dashboard';
import { Experiments } from './pages/Experiments';
import { ExperimentDetail } from './pages/ExperimentDetail';
import { Models } from './pages/Models';
import { ModelDetail } from './pages/ModelDetail';
import { Datasets } from './pages/Datasets';
import { DatasetDetail } from './pages/DatasetDetail';
import { Training } from './pages/Training';
import { Evaluation } from './pages/Evaluation';
import { Settings } from './pages/Settings';
import { Agents } from './pages/Agents';
import { Monitoring } from './pages/Monitoring';
import { Analytics } from './pages/Analytics';
import { Documentation } from './pages/Documentation';

// Hooks and Utils
import { useAppStore } from './store/appStore';
import { useThemeStore } from './store/themeStore';
import { useSystemMetrics } from './hooks/useSystemMetrics';
import { useWebSocket } from './hooks/useWebSocket';
import { ErrorLogger } from './utils/errorLogger';

// Constants
const QUERY_CLIENT_CONFIG = {
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      cacheTime: 1000 * 60 * 10, // 10 minutes
      refetchOnWindowFocus: false,
      retry: 3,
    },
    mutations: {
      retry: 1,
    },
  },
};

const queryClient = new QueryClient(QUERY_CLIENT_CONFIG);

// Theme configuration
const createAppTheme = (mode: 'light' | 'dark', customColors?: any) => {
  return createTheme({
    palette: {
      mode,
      primary: {
        main: mode === 'dark' ? '#90caf9' : '#1976d2',
        light: mode === 'dark' ? '#bbdefb' : '#42a5f5',
        dark: mode === 'dark' ? '#64b5f6' : '#1565c0',
      },
      secondary: {
        main: mode === 'dark' ? '#f48fb1' : '#dc004e',
        light: mode === 'dark' ? '#f8bbd9' : '#f50057',
        dark: mode === 'dark' ? '#f06292' : '#c51162',
      },
      background: {
        default: mode === 'dark' ? '#0a0e27' : '#f5f5f5',
        paper: mode === 'dark' ? '#1e1e1e' : '#ffffff',
      },
      text: {
        primary: mode === 'dark' ? '#ffffff' : '#333333',
        secondary: mode === 'dark' ? '#b0b0b0' : '#666666',
      },
      error: {
        main: mode === 'dark' ? '#f44336' : '#d32f2f',
      },
      warning: {
        main: mode === 'dark' ? '#ff9800' : '#ed6c02',
      },
      info: {
        main: mode === 'dark' ? '#2196f3' : '#0288d1',
      },
      success: {
        main: mode === 'dark' ? '#4caf50' : '#2e7d32',
      },
      ...customColors,
    },
    typography: {
      fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
      h1: {
        fontSize: '2.5rem',
        fontWeight: 600,
        lineHeight: 1.2,
      },
      h2: {
        fontSize: '2rem',
        fontWeight: 600,
        lineHeight: 1.3,
      },
      h3: {
        fontSize: '1.75rem',
        fontWeight: 600,
        lineHeight: 1.4,
      },
      h4: {
        fontSize: '1.5rem',
        fontWeight: 600,
        lineHeight: 1.4,
      },
      h5: {
        fontSize: '1.25rem',
        fontWeight: 600,
        lineHeight: 1.5,
      },
      h6: {
        fontSize: '1rem',
        fontWeight: 600,
        lineHeight: 1.5,
      },
      body1: {
        fontSize: '0.875rem',
        lineHeight: 1.5,
      },
      body2: {
        fontSize: '0.75rem',
        lineHeight: 1.5,
      },
      button: {
        fontSize: '0.875rem',
        fontWeight: 500,
        textTransform: 'none',
      },
    },
    shape: {
      borderRadius: 12,
    },
    spacing: 8,
    components: {
      MuiButton: {
        styleOverrides: {
          root: {
            borderRadius: 8,
            textTransform: 'none',
            fontWeight: 500,
            boxShadow: 'none',
            '&:hover': {
              boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
            },
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: 16,
            boxShadow: mode === 'dark' 
              ? '0 4px 20px rgba(0,0,0,0.3)' 
              : '0 4px 20px rgba(0,0,0,0.1)',
            border: mode === 'dark' ? '1px solid rgba(255,255,255,0.1)' : 'none',
          },
        },
      },
      MuiPaper: {
        styleOverrides: {
          root: {
            backgroundImage: 'none',
          },
        },
      },
      MuiAppBar: {
        styleOverrides: {
          root: {
            backgroundColor: mode === 'dark' ? '#1e1e1e' : '#ffffff',
            color: mode === 'dark' ? '#ffffff' : '#333333',
            boxShadow: mode === 'dark' 
              ? '0 1px 3px rgba(0,0,0,0.3)' 
              : '0 1px 3px rgba(0,0,0,0.1)',
          },
        },
      },
      MuiDrawer: {
        styleOverrides: {
          paper: {
            backgroundColor: mode === 'dark' ? '#1a1a1a' : '#fafafa',
            borderRight: mode === 'dark' ? '1px solid rgba(255,255,255,0.1)' : '1px solid rgba(0,0,0,0.1)',
          },
        },
      },
    },
  });
};

// Main App Component
const App: React.FC = () => {
  const [isInitialized, setIsInitialized] = useState(false);
  const [initError, setInitError] = useState<string | null>(null);
  
  const { 
    setInitialized, 
    setApiConnected, 
    addNotification,
    config,
    updateConfig
  } = useAppStore();
  
  const { 
    themeMode, 
    customColors,
    setThemeMode 
  } = useThemeStore();

  const theme = createAppTheme(themeMode, customColors);

  // Initialize application
  useEffect(() => {
    const initializeApp = async () => {
      try {
        console.log('Initializing Reward Modeling Platform...');
        
        // Initialize the Tauri backend
        await invoke('initialize_app');
        
        // Load application configuration
        const appConfig = await invoke('get_app_config');
        if (appConfig) {
          updateConfig(appConfig);
        }
        
        // Check API health if configured
        if (config?.api?.enable_api) {
          try {
            const isHealthy = await invoke('check_api_health');
            setApiConnected(isHealthy);
          } catch (error) {
            console.warn('API health check failed:', error);
            setApiConnected(false);
          }
        }
        
        setInitialized(true);
        setIsInitialized(true);
        
        addNotification({
          type: 'success',
          title: 'Application Initialized',
          message: 'Reward Modeling Platform is ready to use',
        });
        
        console.log('Application initialized successfully');
        
      } catch (error) {
        console.error('Failed to initialize application:', error);
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        setInitError(errorMessage);
        
        addNotification({
          type: 'error',
          title: 'Initialization Failed',
          message: errorMessage,
        });
        
        // Log error for debugging
        ErrorLogger.logError(new Error(`App initialization failed: ${errorMessage}`));
      }
    };

    initializeApp();
  }, [setInitialized, setApiConnected, addNotification, config?.api?.enable_api, updateConfig]);

  // System metrics monitoring
  useSystemMetrics({
    enabled: isInitialized && config?.monitoring?.enable_monitoring,
    interval: config?.monitoring?.collection_interval || 5000,
  });

  // WebSocket connection for real-time updates
  useWebSocket({
    enabled: isInitialized && config?.api?.enable_api,
    url: config?.websocket?.url,
  });

  // Handle theme changes from system
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    
    const handleThemeChange = (e: MediaQueryListEvent) => {
      if (config?.ui?.theme === 'system') {
        setThemeMode(e.matches ? 'dark' : 'light');
      }
    };

    if (config?.ui?.theme === 'system') {
      setThemeMode(mediaQuery.matches ? 'dark' : 'light');
      mediaQuery.addEventListener('change', handleThemeChange);
      
      return () => {
        mediaQuery.removeEventListener('change', handleThemeChange);
      };
    }
  }, [config?.ui?.theme, setThemeMode]);

  // Error handling for the entire app
  const handleError = (error: Error, errorInfo: any) => {
    console.error('Application error:', error, errorInfo);
    ErrorLogger.logError(error, errorInfo);
    
    addNotification({
      type: 'error',
      title: 'Application Error',
      message: error.message,
    });
  };

  // Show loading screen during initialization
  if (!isInitialized && !initError) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <LoadingScreen message="Initializing Reward Modeling Platform..." />
      </ThemeProvider>
    );
  }

  // Show error screen if initialization failed
  if (initError) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <ErrorFallback 
          error={new Error(initError)} 
          resetErrorBoundary={() => {
            setInitError(null);
            setIsInitialized(false);
          }}
        />
      </ThemeProvider>
    );
  }

  return (
    <ErrorBoundary FallbackComponent={ErrorFallback} onError={handleError}>
      <QueryClientProvider client={queryClient}>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          <SnackbarProvider 
            maxSnack={5}
            anchorOrigin={{
              vertical: 'top',
              horizontal: 'right',
            }}
            autoHideDuration={5000}
          >
            <Router>
              <AppLayout>
                <Routes>
                  {/* Main Dashboard */}
                  <Route path="/" element={<Navigate to="/dashboard" replace />} />
                  <Route path="/dashboard" element={<Dashboard />} />
                  
                  {/* Experiments */}
                  <Route path="/experiments" element={<Experiments />} />
                  <Route path="/experiments/:id" element={<ExperimentDetail />} />
                  
                  {/* Models */}
                  <Route path="/models" element={<Models />} />
                  <Route path="/models/:id" element={<ModelDetail />} />
                  
                  {/* Datasets */}
                  <Route path="/datasets" element={<Datasets />} />
                  <Route path="/datasets/:id" element={<DatasetDetail />} />
                  
                  {/* Training */}
                  <Route path="/training" element={<Training />} />
                  
                  {/* Evaluation */}
                  <Route path="/evaluation" element={<Evaluation />} />
                  
                  {/* Agents */}
                  <Route path="/agents" element={<Agents />} />
                  
                  {/* Monitoring */}
                  <Route path="/monitoring" element={<Monitoring />} />
                  
                  {/* Analytics */}
                  <Route path="/analytics" element={<Analytics />} />
                  
                  {/* Documentation */}
                  <Route path="/docs" element={<Documentation />} />
                  
                  {/* Settings */}
                  <Route path="/settings" element={<Settings />} />
                  
                  {/* Catch all route */}
                  <Route path="*" element={<Navigate to="/dashboard" replace />} />
                </Routes>
              </AppLayout>
              
              {/* Global Notification Manager */}
              <NotificationManager />
            </Router>
          </SnackbarProvider>
          
          {/* React Query DevTools (only in development) */}
          {process.env.NODE_ENV === 'development' && (
            <ReactQueryDevtools initialIsOpen={false} />
          )}
        </ThemeProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  );
};

export default App; 