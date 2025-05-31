import React, { useState } from 'react';
import { FileUploader } from './FileUploader';
import { ProcessingStatus } from './ProcessingStatus';
import { LawsList } from './LawsList';
import { mockProcessDocument, processDocument } from '../utils/documentProcessor';
import { Law } from '../types/Law';

export const DocumentAnalyzer: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [laws, setLaws] = useState<Law[]>([]);

  const handleFileUpload = (uploadedFile: File) => {
    setFile(uploadedFile);
    setLaws([]);
    setCurrentStep(0);
  };

  const handleProcess = async () => {
    if (!file) return;

    setIsProcessing(true);
    setCurrentStep(1);

    try {
      // In a real application, this would be an API call to process the document
      //const result = await mockProcessDocument(
      const result = await processDocument(
        file,
        (step) => setCurrentStep(step)
      );

      setLaws(result);
    } catch (error) {
      console.error('Error processing document:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleClearFile = () => {
    setFile(null);
    setLaws([]);
    setCurrentStep(0);
  };

  return (
    <div className="max-w-5xl mx-auto">
      <div className="mb-8 bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-800">Sube tu documento</h2>
        <FileUploader
          onFileUpload={handleFileUpload}
          onClearFile={handleClearFile}
          file={file}
        />

        {file && (
          <div className="mt-4 flex justify-end">
            <button
              onClick={handleProcess}
              disabled={isProcessing}
              className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors duration-200 disabled:bg-blue-400 disabled:cursor-not-allowed flex items-center"
            >
              {isProcessing ? (
                <>
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white\" xmlns="http://www.w3.org/2000/svg\" fill="none\" viewBox="0 0 24 24">
                    <circle className="opacity-25\" cx="12\" cy="12\" r="10\" stroke="currentColor\" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Procesando...
                </>
              ) : (
                'Procesar documento'
              )}
            </button>
          </div>
        )}
      </div>

      {(isProcessing || currentStep > 0) && (
        <div className="mb-8 bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4 text-gray-800">Estado del procesamiento</h2>
          <ProcessingStatus currentStep={currentStep} isProcessing={isProcessing} />
        </div>
      )}

      {laws.length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4 text-gray-800">Leyes encontradas</h2>
          <LawsList laws={laws} />
        </div>
      )}
    </div>
  );
};