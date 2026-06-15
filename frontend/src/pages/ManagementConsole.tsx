import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { getDomains, createDomain, getTemplates, deleteTemplate } from '../api/domains';
import { domainTagClass, domainTagBase, domainPillClass } from '../lib/domains';
import EditDomainsModal from '../components/EditDomainsModal';
import type { Domain, Template } from '../api/types';

export default function ManagementConsole() {
  const [domains, setDomains] = useState<Domain[]>([]);
  const [templates, setTemplates] = useState<Template[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeDomain, setActiveDomain] = useState<number | 'all'>('all');
  const [newDomainName, setNewDomainName] = useState('');
  const [showAddDomain, setShowAddDomain] = useState(false);
  const [showEditDomains, setShowEditDomains] = useState(false);
  const [search, setSearch] = useState('');
  const [savingDomain, setSavingDomain] = useState(false);
  const newDomainRef = useRef<HTMLInputElement>(null);

  const fetchAll = () =>
    Promise.all([getDomains(), getTemplates()])
      .then(([d, t]) => { setDomains(d); setTemplates(t); })
      .finally(() => setLoading(false));

  useEffect(() => { fetchAll(); }, []);

  useEffect(() => {
    if (showAddDomain) newDomainRef.current?.focus();
  }, [showAddDomain]);

  const handleCreateDomain = async () => {
    const name = newDomainName.trim();
    if (!name) return;
    setSavingDomain(true);
    const d = await createDomain({ name }).catch(() => null);
    setSavingDomain(false);
    if (d) {
      setDomains((prev) => [...prev, d]);
      setNewDomainName('');
      setShowAddDomain(false);
    }
  };

  const handleDeleteTemplate = async (id: number) => {
    await deleteTemplate(id).catch(() => null);
    setTemplates((prev) => prev.filter((t) => t.id !== id));
  };

  const handleEditDomainsClose = () => {
    setShowEditDomains(false);
    // Re-fetch so colors, order, and deletions are reflected
    getDomains().then((d) => {
      setDomains(d);
      const validIds = new Set(d.map((dom) => dom.id));
      setTemplates((prev) => prev.map((t) =>
        t.domain_id && !validIds.has(t.domain_id) ? { ...t, domain_id: null } : t
      ));
      if (activeDomain !== 'all' && !validIds.has(activeDomain as number)) {
        setActiveDomain('all');
      }
    });
  };

  const domainObj = (id: number | null) => id ? domains.find((d) => d.id === id) ?? null : null;
  const visibleTemplates = templates
    .filter((t) => activeDomain === 'all' || t.domain_id === activeDomain)
    .filter((t) => !search || t.name.toLowerCase().includes(search.toLowerCase()));

  return (
    <main className="flex-1 w-full flex flex-col min-h-screen">
      {showEditDomains && (
        <EditDomainsModal domains={domains} onClose={handleEditDomainsClose} />
      )}

      <header className="flex justify-between items-center px-margin-mobile md:px-margin-desktop py-space-6 max-w-container-max mx-auto w-full sticky top-0 bg-surface/90 backdrop-blur-md z-30 border-b border-outline-variant/30">
        <div>
          <h2 className="font-headline-lg-mobile md:font-headline-lg text-headline-lg-mobile md:text-headline-lg text-on-surface tracking-tight">
            Templates
          </h2>
          <p className="font-body-md text-body-md text-on-surface-variant mt-1">
            Prompt configurations and domain classifications.
          </p>
        </div>
      </header>

      <div className="flex-1 px-margin-mobile md:px-margin-desktop py-space-6 max-w-container-max mx-auto w-full space-y-space-8">

        {/* Templates Section */}
        <section className="space-y-space-4">
          <div className="flex items-center justify-between gap-space-3">
            <div className="flex flex-1 items-center bg-surface-container-lowest border border-outline-variant rounded px-3 py-1.5 focus-within:border-primary focus-within:ring-1 focus-within:ring-primary transition-all shadow-sm max-w-xs">
              <span className="material-symbols-outlined text-on-surface-variant text-[18px] mr-2">search</span>
              <input
                className="flex-1 bg-transparent border-none outline-none focus:ring-0 font-body-sm text-body-sm text-on-surface placeholder:text-on-surface-variant"
                placeholder="Search templates…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </div>
            <Link
              to="/management/templates/new/edit"
              className="bg-primary text-on-primary font-label-md text-label-md py-2 px-4 rounded-lg flex items-center gap-2 hover:opacity-90 transition-opacity shadow-sm"
            >
              <span className="material-symbols-outlined text-[18px]">add</span>
              New Template
            </Link>
          </div>

          {/* Domain filter pills */}
          {!loading && (
            <div className="flex flex-wrap items-center gap-2">
              {/* All pill */}
              <button
                onClick={() => setActiveDomain('all')}
                className={`px-3 py-1 font-label-sm text-label-sm border transition-colors ${
                  activeDomain === 'all'
                    ? 'bg-primary text-on-primary border-primary'
                    : 'bg-surface-container-lowest border-outline-variant text-on-surface-variant hover:border-primary hover:text-primary'
                }`}
              >
                All
              </button>

              {/* Domain pills */}
              {domains.map((d) => (
                <button
                  key={d.id}
                  onClick={() => setActiveDomain(activeDomain === d.id ? 'all' : d.id)}
                  className={`px-3 py-1 font-label-sm text-label-sm border transition-colors ${domainPillClass(d.name, d.color, activeDomain === d.id)}`}
                >
                  {d.name}
                </button>
              ))}

              {/* Add domain */}
              {showAddDomain ? (
                <div className="flex items-center gap-2 bg-surface-container-lowest border border-primary/30 px-3 py-1">
                  <input
                    ref={newDomainRef}
                    className="bg-transparent border-none focus:ring-0 font-label-sm text-label-sm text-on-surface w-32 placeholder:text-outline"
                    placeholder="Domain name…"
                    value={newDomainName}
                    onChange={(e) => setNewDomainName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleCreateDomain();
                      if (e.key === 'Escape') { setShowAddDomain(false); setNewDomainName(''); }
                    }}
                  />
                  <button
                    onClick={handleCreateDomain}
                    disabled={!newDomainName.trim() || savingDomain}
                    className="text-primary hover:text-primary/70 transition-colors disabled:opacity-40"
                  >
                    <span className="material-symbols-outlined text-[16px]">check</span>
                  </button>
                  <button
                    onClick={() => { setShowAddDomain(false); setNewDomainName(''); }}
                    className="text-on-surface-variant hover:text-error transition-colors"
                  >
                    <span className="material-symbols-outlined text-[16px]">close</span>
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => setShowAddDomain(true)}
                  className="px-3 py-1 font-label-sm text-label-sm border border-dashed border-outline-variant text-on-surface-variant hover:border-primary hover:text-primary transition-colors flex items-center gap-1"
                >
                  <span className="material-symbols-outlined text-[14px]">add</span>
                  Add Domain
                </button>
              )}

              {/* Edit domains */}
              <button
                onClick={() => setShowEditDomains(true)}
                className="px-3 py-1 font-label-sm text-label-sm border border-outline-variant text-on-surface-variant hover:border-primary hover:text-primary transition-colors flex items-center gap-1"
              >
                <span className="material-symbols-outlined text-[14px]">tune</span>
                Edit Domains
              </button>
            </div>
          )}

          {loading ? (
            <div className="p-space-6 text-center text-on-surface-variant font-body-sm text-body-sm">Loading…</div>
          ) : visibleTemplates.length === 0 ? (
            <div className="bg-surface-container-lowest rounded-lg p-space-8 text-center text-on-surface-variant font-body-sm text-body-sm">
              {activeDomain === 'all' ? 'No templates yet.' : 'No templates for this domain.'}
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-space-4">
              {visibleTemplates.map((t) => {
                const dObj = domainObj(t.domain_id);
                return (
                  <div
                    key={t.id}
                    className="group bg-surface-container-lowest p-space-4 rounded flex flex-col hover:shadow-[0_4px_16px_rgba(0,0,0,0.04)] transition-all relative"
                  >
                    <div className="absolute top-3 right-3 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <Link
                        to={`/management/templates/${t.id}/edit`}
                        className="p-1 text-on-surface-variant hover:text-primary transition-colors"
                        title="Edit"
                      >
                        <span className="material-symbols-outlined text-[18px]">edit</span>
                      </Link>
                      <button
                        onClick={() => handleDeleteTemplate(t.id)}
                        className="p-1 text-on-surface-variant hover:text-error transition-colors"
                        title="Delete"
                      >
                        <span className="material-symbols-outlined text-[18px]">delete</span>
                      </button>
                    </div>

                    {/* Domain tag — top of card */}
                    <div className="mb-space-3">
                      <span className={`${domainTagBase} ${domainTagClass(dObj?.name ?? '—', dObj?.color)}`}>
                        {dObj?.name ?? '—'}
                      </span>
                    </div>

                    <h4 className="font-body-lg font-semibold text-on-surface mb-1 pr-12">{t.name}</h4>
                    <p className="font-body-sm text-body-sm text-on-surface-variant line-clamp-2 mb-space-4 flex-1">
                      {t.description || <span className="italic opacity-50">No description yet</span>}
                    </p>

                    <div className="flex items-center justify-end mt-auto pt-space-2 border-t border-outline-variant/30">
                      <div className="flex gap-1">
                        {t.output_sections.map((s) => (
                          <span key={s} className="text-[10px] font-label-sm text-on-surface-variant bg-surface-container-high rounded px-1.5 py-0.5 capitalize">
                            {s.replace('_', ' ')}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
