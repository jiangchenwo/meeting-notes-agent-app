import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';

export default function Layout() {
  return (
    <div className="flex min-h-screen bg-surface text-on-surface">
      <Sidebar />
      <div className="flex-1 md:ml-64 min-h-screen flex flex-col">
        <Outlet />
      </div>
    </div>
  );
}
