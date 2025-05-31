import React, { useCallback, useState } from 'react';
import { FileText, X, Upload } from 'lucide-react';

interface FileUploaderProps {
  onFileUpload: (file: File) => void;
  onClearFile: () => void;
  file: File | null;
}

export const FileUploader: React.FC<FileUploaderProps> = ({ onFileUpload, onClearFile, file }) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragEnter = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const droppedFile = files[0];
      const fileExtension = droppedFile.name.split('.').pop()?.toLowerCase();
      
      // Only accept .docx and .doc files
      if (fileExtension === 'docx' || fileExtension === 'doc') {
        onFileUpload(droppedFile);
      } else {
        alert('Por favor, sube un documento de Word (.docx o .doc)');
      }
    }
  }, [onFileUpload]);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onFileUpload(files[0]);
    }
  }, [onFileUpload]);

  return (
    <div className="w-full">
      {!file ? (
        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center ${
            isDragging 
              ? 'border-blue-500 bg-blue-50' 
              : 'border-gray-300 hover:border-blue-400 hover:bg-blue-50'
          } transition-colors duration-200 cursor-pointer`}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onClick={() => document.getElementById('file-upload')?.click()}
        >
          <div className="flex flex-col items-center">
            <Upload className="h-12 w-12 text-blue-500 mb-3" />
            <p className="text-lg font-medium text-gray-700">
              Arrastra y suelta tu documento aquí
            </p>
            <p className="text-sm text-gray-500 mt-1">
              o haz clic para seleccionar un archivo
            </p>
            <p className="text-xs text-gray-400 mt-2">
              Formatos soportados: .docx, .doc
            </p>
            <input
              id="file-upload"
              type="file"
              className="hidden"
              onChange={handleFileChange}
              accept=".doc,.docx"
            />
          </div>
        </div>
      ) : (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-start">
          <div className="bg-blue-100 p-3 rounded-md mr-4">
            <FileText className="h-8 w-8 text-blue-600" />
          </div>
          <div className="flex-1">
            <div className="flex justify-between items-start">
              <div>
                <p className="font-medium text-gray-800 truncate max-w-xs md:max-w-md">{file.name}</p>
                <p className="text-sm text-gray-500">
                  {(file.size / 1024).toFixed(2)} KB • {file.type || 'documento Word'}
                </p>
              </div>
              <button 
                onClick={onClearFile}
                className="text-gray-500 hover:text-red-500 p-1 rounded-full hover:bg-red-50 transition-colors"
                aria-label="Eliminar archivo"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};