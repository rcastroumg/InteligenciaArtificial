import React, { useState } from 'react';
import { LawCard } from './LawCard';
import { Law } from '../types/Law';
import { Search } from 'lucide-react';

interface LawsListProps {
  laws: Law[];
}

export const LawsList: React.FC<LawsListProps> = ({ laws }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterActive, setFilterActive] = useState<boolean | null>(null);

  const filteredLaws = laws.filter(law => {
    const matchesSearch = law.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
                          law.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          law.id.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesActiveFilter = filterActive === null || law.isActive === filterActive;
    
    return matchesSearch && matchesActiveFilter;
  });

  return (
    <div>
      <div className="mb-6 flex flex-col md:flex-row gap-4">
        <div className="relative flex-1">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Search className="h-5 w-5 text-gray-400" />
          </div>
          <input
            type="text"
            className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
            placeholder="Buscar leyes..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        <div className="flex items-center gap-4">
          <span className="text-sm text-gray-700">Estado:</span>
          <div className="flex gap-2">
            <button
              className={`px-3 py-1 text-sm rounded-md ${
                filterActive === null 
                  ? 'bg-blue-100 text-blue-800' 
                  : 'bg-gray-100 text-gray-800 hover:bg-gray-200'
              }`}
              onClick={() => setFilterActive(null)}
            >
              Todos
            </button>
            <button
              className={`px-3 py-1 text-sm rounded-md ${
                filterActive === true 
                  ? 'bg-green-100 text-green-800' 
                  : 'bg-gray-100 text-gray-800 hover:bg-gray-200'
              }`}
              onClick={() => setFilterActive(true)}
            >
              Activas
            </button>
            <button
              className={`px-3 py-1 text-sm rounded-md ${
                filterActive === false 
                  ? 'bg-red-100 text-red-800' 
                  : 'bg-gray-100 text-gray-800 hover:bg-gray-200'
              }`}
              onClick={() => setFilterActive(false)}
            >
              Inactivas
            </button>
          </div>
        </div>
      </div>

      {filteredLaws.length === 0 ? (
        <div className="text-center py-8 text-gray-500">
          No se encontraron leyes que coincidan con los criterios de búsqueda.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredLaws.map((law) => (
            <LawCard key={law.id} law={law} />
          ))}
        </div>
      )}
    </div>
  );
};