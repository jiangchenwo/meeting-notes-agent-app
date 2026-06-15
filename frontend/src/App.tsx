import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import MainPage from './pages/MainPage';
import ProjectsOverview from './pages/ProjectsOverview';
import ProjectDetail from './pages/ProjectDetail';
import AudioManagement from './pages/AudioManagement';
import ManagementConsole from './pages/ManagementConsole';
import EditTemplate from './pages/EditTemplate';
import Settings from './pages/Settings';

// Testing Lab is dev-only; excluded from production builds via VITE_ENABLE_LAB=false.
const LAB_ENABLED = import.meta.env.VITE_ENABLE_LAB !== 'false';
const TestingLab = lazy(() => import('./pages/TestingLab'));

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<MainPage />} />
          <Route path="projects" element={<ProjectsOverview />} />
          <Route path="projects/:id" element={<ProjectDetail />} />
          <Route path="notes/:noteId" element={<AudioManagement />} />
          <Route path="projects/:id/notes/:noteId" element={<AudioManagement />} />
          <Route path="management" element={<ManagementConsole />} />
          <Route path="management/templates/:templateId/edit" element={<EditTemplate />} />
          <Route path="settings" element={<Settings />} />
          {LAB_ENABLED && (
            <Route
              path="lab"
              element={
                <Suspense fallback={null}>
                  <TestingLab />
                </Suspense>
              }
            />
          )}
          <Route path="archive" element={<div className="p-8 text-on-surface-variant">Archive — coming soon</div>} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
