import { createContext, useContext, useState, ReactNode } from 'react';

interface GuideSection {
  id: string;
  title: string;
  description: string;
  completed: boolean;
  icon: string;
}

interface GuideContextType {
  sections: GuideSection[];
  currentSection: string;
  setCurrentSection: (id: string) => void;
  markCompleted: (id: string) => void;
  progress: number;
}

const GuideContext = createContext<GuideContextType | undefined>(undefined);

const defaultSections: GuideSection[] = [
  {
    id: 'overview',
    title: 'Overview & Introduction',
    description: 'Understanding conversational AI systems and their components',
    completed: false,
    icon: 'Brain'
  },
  {
    id: 'core-components',
    title: 'Core Components',
    description: 'LLMs, tokenizers, inference systems, and architecture',
    completed: false,
    icon: 'Cpu'
  },
  {
    id: 'data-management',
    title: 'Data Management',
    description: 'Collection, preprocessing, and ethical handling of training data',
    completed: false,
    icon: 'Database'
  },
  {
    id: 'frameworks',
    title: 'Open Source Frameworks',
    description: 'Hugging Face, LLaMA, Mistral, Falcon, and other tools',
    completed: false,
    icon: 'Code'
  },
  {
    id: 'infrastructure',
    title: 'Infrastructure & Resources',
    description: 'Hardware requirements and cost-effective cloud solutions',
    completed: false,
    icon: 'Cloud'
  },
  {
    id: 'fine-tuning',
    title: 'Fine-tuning & Customization',
    description: 'Adapting models for specific use cases and domains',
    completed: false,
    icon: 'Zap'
  },
  {
    id: 'safety',
    title: 'Safety & Ethics',
    description: 'Moderation, alignment, and bias prevention techniques',
    completed: false,
    icon: 'Shield'
  },
  {
    id: 'api-integration',
    title: 'API & Integration',
    description: 'Building APIs and integrating into applications',
    completed: false,
    icon: 'Plug'
  },
  {
    id: 'scaling',
    title: 'Scaling & Maintenance',
    description: 'Production deployment and ongoing maintenance',
    completed: false,
    icon: 'TrendingUp'
  }
];

export function GuideProvider({ children }: { children: ReactNode }) {
  const [sections, setSections] = useState<GuideSection[]>(defaultSections);
  const [currentSection, setCurrentSection] = useState('overview');

  const markCompleted = (id: string) => {
    setSections(prev => prev.map(section => 
      section.id === id ? { ...section, completed: true } : section
    ));
  };

  const progress = Math.round((sections.filter(s => s.completed).length / sections.length) * 100);

  return (
    <GuideContext.Provider value={{
      sections,
      currentSection,
      setCurrentSection,
      markCompleted,
      progress
    }}>
      {children}
    </GuideContext.Provider>
  );
}

export function useGuide() {
  const context = useContext(GuideContext);
  if (!context) {
    throw new Error('useGuide must be used within a GuideProvider');
  }
  return context;
}
