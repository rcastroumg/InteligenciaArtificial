import React from 'react';
import { CheckCircle, Clock, AlertCircle, Loader } from 'lucide-react';

interface ProcessingStatusProps {
  currentStep: number;
  isProcessing: boolean;
}

const steps = [
  { id: 1, name: 'Analizando archivo', description: 'Extrayendo contenido del documento' },
  { id: 2, name: 'Extrayendo leyes', description: 'Identificando referencias legales' },
  { id: 3, name: 'Verificando estado', description: 'Comprobando vigencia y reemplazos' },
  { id: 4, name: 'Completado', description: 'Análisis finalizado' }
];

export const ProcessingStatus: React.FC<ProcessingStatusProps> = ({ currentStep, isProcessing }) => {
  return (
    <div className="space-y-4">
      {steps.map((step) => {
        let status: 'pending' | 'current' | 'completed' | 'error' = 'pending';
        
        if (step.id < currentStep) {
          status = 'completed';
        } else if (step.id === currentStep) {
          status = isProcessing ? 'current' : 'completed';
        }
        
        return (
          <div key={step.id} className="flex items-start">
            <div className="flex-shrink-0 mr-3">
              {status === 'pending' && (
                <Clock className="h-6 w-6 text-gray-400" />
              )}
              {status === 'current' && (
                <Loader className="h-6 w-6 text-blue-500 animate-spin" />
              )}
              {status === 'completed' && (
                <CheckCircle className="h-6 w-6 text-green-500" />
              )}
              {status === 'error' && (
                <AlertCircle className="h-6 w-6 text-red-500" />
              )}
            </div>
            <div className="flex-1">
              <div className="flex items-center">
                <h3 className={`font-medium ${
                  status === 'pending' ? 'text-gray-500' :
                  status === 'current' ? 'text-blue-600' :
                  status === 'completed' ? 'text-green-600' :
                  'text-red-600'
                }`}>
                  {step.name}
                </h3>
                {status === 'current' && (
                  <div className="ml-2 text-sm bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full">
                    En progreso
                  </div>
                )}
              </div>
              <p className="text-sm text-gray-500">{step.description}</p>
            </div>
          </div>
        );
      })}
    </div>
  );
};