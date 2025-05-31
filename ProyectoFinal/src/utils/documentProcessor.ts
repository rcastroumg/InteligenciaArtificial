import { Law } from '../types/Law';

// Mock function to simulate document processing
// In a real application, this would be replaced with actual document parsing logic
export const mockProcessDocument = async (
  file: File,
  onStepChange: (step: number) => void
): Promise<Law[]> => {
  // Step 1: Analyzing file
  await simulateProcessingStep(1500);
  onStepChange(1);

  // Step 2: Extracting laws
  await simulateProcessingStep(2000);
  onStepChange(2);

  // Step 3: Verifying law status
  await simulateProcessingStep(1800);
  onStepChange(3);

  // Step 4: Complete
  onStepChange(4);

  // Return mock data
  return generateMockLaws();
};

const simulateProcessingStep = (ms: number): Promise<void> => {
  return new Promise(resolve => setTimeout(resolve, ms));
};

const generateMockLaws = (): Law[] => {
  return [
    {
      id: "LEY-20255",
      name: "Ley de Reforma Previsional",
      description: "Establece la Reforma Previsional, creando un sistema de pensiones solidarias de vejez e invalidez, complementario del sistema de pensiones a que se refiere el decreto ley N° 3.500, de 1980.",
      isActive: true,
      lastUpdate: "2022-03-15"
    },
    {
      id: "DL-3500",
      name: "Decreto Ley 3500",
      description: "Establece el Sistema de Pensiones basado en la Capitalización Individual administrado por las Administradoras de Fondos de Pensiones (AFP).",
      isActive: true,
      lastUpdate: "2021-11-20"
    },
    {
      id: "LEY-18290",
      name: "Ley de Tránsito",
      description: "Fija el texto refundido, coordinado y sistematizado de la Ley de Tránsito. Regula el tránsito en las vías públicas, los vehículos, los conductores, el transporte de pasajeros y de carga, y las condiciones de seguridad.",
      isActive: true,
      lastUpdate: "2022-01-05"
    },
    {
      id: "LEY-19496",
      name: "Ley de Protección al Consumidor",
      description: "Establece normas sobre protección de los derechos de los consumidores. Regula las relaciones entre proveedores y consumidores, establece los derechos y deberes de los consumidores y el procedimiento aplicable en estas materias.",
      isActive: true,
      lastUpdate: "2021-09-30"
    },
    {
      id: "LEY-16744",
      name: "Ley de Accidentes del Trabajo",
      description: "Establece normas sobre accidentes del trabajo y enfermedades profesionales. Crea un seguro social obligatorio contra riesgos de accidentes del trabajo y enfermedades profesionales.",
      isActive: true,
      lastUpdate: "2021-12-10"
    },
    {
      id: "LEY-18834",
      name: "Estatuto Administrativo",
      description: "Aprueba el Estatuto Administrativo que regula las relaciones entre el Estado y el personal de los Ministerios, Intendencias, Gobernaciones y servicios públicos centralizados y descentralizados.",
      isActive: true,
      lastUpdate: "2022-02-28"
    },
    {
      id: "LEY-14908",
      name: "Ley de Abandono de Familia y Pago de Pensiones Alimenticias",
      description: "Fija el texto definitivo de la Ley N° 5.750, sobre Abandono de Familia y Pago de Pensiones Alimenticias. Establece procedimientos para el cobro y cumplimiento de la obligación de proporcionar alimentos.",
      isActive: false,
      replacementLaw: "LEY-21389",
      lastUpdate: "2021-06-18"
    },
    {
      id: "LEY-21389",
      name: "Ley que crea el Registro Nacional de Deudores de Pensiones de Alimentos",
      description: "Crea el Registro Nacional de Deudores de Pensiones de Alimentos y modifica diversos cuerpos legales para perfeccionar el sistema de pago de las pensiones de alimentos.",
      isActive: true,
      lastUpdate: "2022-05-10"
    },
    {
      id: "DFL-1",
      name: "Código del Trabajo",
      description: "Fija el texto refundido, coordinado y sistematizado del Código del Trabajo. Regula las relaciones laborales entre los empleadores y los trabajadores, estableciendo los derechos y obligaciones de ambas partes.",
      isActive: true,
      lastUpdate: "2022-04-05"
    },
    {
      id: "LEY-10336",
      name: "Ley de Organización y Atribuciones de la Contraloría General de la República",
      description: "Determina la organización y atribuciones de la Contraloría General de la República, órgano constitucional autónomo encargado del control de la legalidad de los actos de la Administración.",
      isActive: true,
      lastUpdate: "2021-10-15"
    },
    {
      id: "LEY-18575",
      name: "Ley Orgánica Constitucional de Bases Generales de la Administración del Estado",
      description: "Establece las bases generales de la Administración del Estado. Regula la organización y funcionamiento de la Administración del Estado y consagra los principios que la rigen.",
      isActive: true,
      lastUpdate: "2022-01-20"
    },
    {
      id: "LEY-19886",
      name: "Ley de Compras Públicas",
      description: "Ley de Bases sobre Contratos Administrativos de Suministro y Prestación de Servicios. Regula los contratos que celebre la Administración del Estado para el suministro de bienes muebles y servicios.",
      isActive: false,
      replacementLaw: "LEY-21559",
      lastUpdate: "2023-06-15"
    }
  ];
};



export const processDocument = async (
  file: File,
  onStepChange: (step: number) => void
): Promise<Law[]> => {
  // Paso 1: Analizando archivo
  onStepChange(1);

  // Crear FormData y agregar el archivo
  const formData = new FormData();
  formData.append('Contenido', file);

  // Paso 2: Enviando al API
  onStepChange(2);

  try {
    const response = await fetch('http://localhost:8002/api/utilidades/iaReadDocument', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('Error en la respuesta del servidor');
    }

    // Paso 3: Procesando respuesta
    onStepChange(3);

    const data = await response.json();

    // Paso 4: Completo
    onStepChange(4);

    // Suponiendo que el API retorna un array de leyes en data.laws
    return data as Law[];
  } catch (error) {
    onStepChange(0);
    throw error;
  }
};