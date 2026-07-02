import { NavLink, useNavigate } from 'react-router-dom';

// Testing Lab is dev-only; excluded from production builds via VITE_ENABLE_LAB=false.
const LAB_ENABLED = import.meta.env.VITE_ENABLE_LAB !== 'false';

const navItems = [
  { to: '/', icon: 'home', label: 'Main Page', exact: true },
  { to: '/projects', icon: 'folder', label: 'Projects' },
  { to: '/management', icon: 'dashboard_customize', label: 'Templates' },
  ...(LAB_ENABLED ? [{ to: '/lab', icon: 'science', label: 'Testing Lab' }] : []),
  { to: '/settings', icon: 'settings', label: 'Settings' },
  { to: '/archive', icon: 'archive', label: 'Archive' },
];

export default function Sidebar() {
  const navigate = useNavigate();
  return (
    <aside className="hidden md:flex fixed left-0 top-0 h-screen w-64 flex-col p-4 border-r border-outline-variant bg-surface-container-lowest z-40">
      <div className="mb-space-8 px-space-2 mt-space-2">
        <h1 className="font-display text-headline-md font-bold text-primary tracking-tight leading-none">
          Meeting Notes
        </h1>
        <p className="text-on-surface-variant font-body-sm text-body-sm mt-1">Local-first Agent</p>
      </div>

      <div className="px-space-4 mb-space-6">
        <button
          onClick={() => navigate('/', { state: { openUpload: true } })}
          className="w-full bg-primary text-on-primary font-label-md text-label-md px-4 py-2 rounded-DEFAULT hover:opacity-90 transition-opacity flex items-center justify-center gap-2 shadow-sm"
        >
          <span className="material-symbols-outlined text-[18px]">mic</span>
          New Recording
        </button>
      </div>

      <nav className="flex-1 space-y-1" aria-label="Main Navigation">
        {navItems.map(({ to, icon, label, exact }) => (
          <NavLink
            key={to}
            to={to}
            end={exact}
            className={({ isActive }) =>
              isActive
                ? 'text-primary font-bold border-r-2 border-primary bg-surface-container-low flex items-center gap-space-3 px-space-4 py-space-2 rounded-l-DEFAULT'
                : 'text-on-surface-variant hover:text-primary hover:bg-surface-container-low transition-all flex items-center gap-space-3 px-space-4 py-space-2 rounded-l-DEFAULT'
            }
          >
            <span className="material-symbols-outlined text-[20px]">{icon}</span>
            <span className="font-label-md text-label-md">{label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="border-t border-outline-variant pt-space-4 mt-space-4 space-y-1">
        <a
          href="https://github.com/jiangchenwo/meeting-notes-agent-app"
          target="_blank"
          rel="noopener noreferrer"
          className="text-on-surface-variant hover:text-primary hover:bg-surface-container-low transition-all flex items-center gap-space-3 px-space-4 py-space-2 rounded-DEFAULT"
        >
          <span className="material-symbols-outlined text-[20px]">help</span>
          <span className="font-label-md text-label-md">Support</span>
        </a>
        <a
          href="#"
          className="text-on-surface-variant hover:text-primary hover:bg-surface-container-low transition-all flex items-center gap-space-3 px-space-4 py-space-2 rounded-DEFAULT"
        >
          <span className="material-symbols-outlined text-[20px]">logout</span>
          <span className="font-label-md text-label-md">Log Out</span>
        </a>
      </div>
    </aside>
  );
}
