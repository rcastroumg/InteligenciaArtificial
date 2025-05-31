export interface Law {
  id: string;
  name: string;
  description: string;
  article?: string;
  isActive: boolean;
  replacementLaw?: string;
  lastUpdate?: string;
  references?: string[];
}