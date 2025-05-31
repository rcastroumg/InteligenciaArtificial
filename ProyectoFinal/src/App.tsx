import React from 'react';
import { DocumentAnalyzer } from './components/DocumentAnalyzer';

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-blue-900 text-white py-4 px-6 shadow-md">
        <div className="container mx-auto">
          <h1 className="text-2xl font-bold">LegaLyzer</h1>
          <p className="text-blue-200">Análisis inteligente de documentos legales</p>
        </div>
      </header>
      
      <main className="container mx-auto py-8 px-4">
        <DocumentAnalyzer />
      </main>
      
      <footer className="bg-gray-100 py-4 border-t border-gray-200">
        <div className="container mx-auto px-4 text-center text-gray-500 text-sm">
          © {new Date().getFullYear()} LegaLyzer. Todos los derechos reservados.
        </div>
      </footer>
    </div>
  );
}

export default App;