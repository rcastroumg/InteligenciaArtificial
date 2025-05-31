import React, { useState } from 'react';
import { Law } from '../types/Law';
import { Check, X, ChevronDown, ChevronUp, RefreshCw } from 'lucide-react';

interface LawCardProps {
  law: Law;
}

export const LawCard: React.FC<LawCardProps> = ({ law }) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-sm hover:shadow-md transition-shadow duration-200 overflow-hidden">
      <div className="p-4">
        <div className="flex justify-between items-start mb-2">
          <h3 className="text-lg font-semibold text-gray-800 line-clamp-1 flex">{law.name}</h3>
          <div className={`px-2 py-1 text-xs font-medium rounded-full ${law.isActive
            ? 'bg-green-100 text-green-800'
            : 'bg-red-100 text-red-800'
            }`}>
            {law.isActive ? 'Activa' : 'Inactiva'}
          </div>
        </div>

        <p className="text-xs text-gray-500 mb-2">Articulo: {law.article}</p>

        <p className={`text-gray-600 text-sm ${expanded ? '' : 'line-clamp-3'}`}>
          {law.description}
        </p>

        {law.description.length > 150 && (
          <button
            className="mt-2 text-blue-600 text-sm flex items-center hover:text-blue-800"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? (
              <>
                <ChevronUp className="h-4 w-4 mr-1" />
                Mostrar menos
              </>
            ) : (
              <>
                <ChevronDown className="h-4 w-4 mr-1" />
                Mostrar más
              </>
            )}
          </button>
        )}
      </div>

      <div className="bg-gray-50 px-4 py-3 border-t border-gray-200">
        <div className="flex items-center text-sm">
          <div className="flex items-center mr-4">
            <span className="mr-1 text-gray-500">Estado:</span>
            {law.isActive ? (
              <span className="flex items-center text-green-600">
                <Check className="h-4 w-4 mr-1" />
                Vigente
              </span>
            ) : (
              <span className="flex items-center text-red-600">
                <X className="h-4 w-4 mr-1" />
                Derogada
              </span>
            )}
          </div>

          {law.replacementLaw && (
            <div className="flex items-center text-sm">
              <span className="mr-1 text-gray-500">Reemplazada por:</span>
              <span className="flex items-center text-blue-600">
                <RefreshCw className="h-4 w-4 mr-1" />
                {law.replacementLaw}
              </span>
            </div>
          )}
        </div>

        {law.lastUpdate && (
          <div className="mt-2 text-xs text-gray-500">
            Última actualización: {new Date(law.lastUpdate).toLocaleDateString()}
          </div>
        )}
      </div>
    </div>
  );
};