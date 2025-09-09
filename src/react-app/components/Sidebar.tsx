import { useNavigate } from 'react-router';
import { CheckCircle, Circle, Brain, Cpu, Database, Code, Cloud, Zap, Shield, Plug, TrendingUp } from 'lucide-react';
import { useGuide } from '@/react-app/contexts/GuideContext';

const iconMap = {
  Brain,
  Cpu,
  Database,
  Code,
  Cloud,
  Zap,
  Shield,
  Plug,
  TrendingUp
};

export default function Sidebar() {
  const navigate = useNavigate();
  const { sections, currentSection, setCurrentSection } = useGuide();

  const handleSectionClick = (sectionId: string) => {
    setCurrentSection(sectionId);
    navigate(`/guide/${sectionId}`);
  };

  return (
    <aside className="w-80 bg-white border-r border-slate-200 sticky top-[73px] h-[calc(100vh-73px)] overflow-y-auto">
      <div className="p-6">
        <h2 className="text-lg font-semibold text-slate-900 mb-6">Guide Sections</h2>
        <nav className="space-y-2">
          {sections.map((section, index) => {
            const Icon = iconMap[section.icon as keyof typeof iconMap] || Brain;
            const isActive = currentSection === section.id;
            
            return (
              <button
                key={section.id}
                onClick={() => handleSectionClick(section.id)}
                className={`w-full flex items-start space-x-3 p-4 rounded-lg text-left transition-all duration-200 group ${
                  isActive 
                    ? 'bg-blue-50 border-l-4 border-blue-500' 
                    : 'hover:bg-slate-50 border-l-4 border-transparent'
                }`}
              >
                <div className="flex-shrink-0 pt-0.5">
                  {section.completed ? (
                    <CheckCircle className="w-5 h-5 text-green-500" />
                  ) : (
                    <Circle className={`w-5 h-5 ${isActive ? 'text-blue-500' : 'text-slate-400'}`} />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-2 mb-1">
                    <Icon className={`w-4 h-4 ${isActive ? 'text-blue-500' : 'text-slate-500'}`} />
                    <span className={`text-sm font-medium ${isActive ? 'text-blue-900' : 'text-slate-900'}`}>
                      {index + 1}. {section.title}
                    </span>
                  </div>
                  <p className={`text-xs leading-relaxed ${isActive ? 'text-blue-700' : 'text-slate-600'}`}>
                    {section.description}
                  </p>
                </div>
              </button>
            );
          })}
        </nav>
      </div>
    </aside>
  );
}
