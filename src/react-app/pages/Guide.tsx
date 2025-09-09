import { useParams, useNavigate } from 'react-router';
import { useEffect } from 'react';
import { ChevronLeft, CheckCircle } from 'lucide-react';
import { useGuide } from '@/react-app/contexts/GuideContext';
import Sidebar from '@/react-app/components/Sidebar';
import SectionContent from '@/react-app/components/SectionContent';
import ProgressBar from '@/react-app/components/ProgressBar';

export default function Guide() {
  const { section } = useParams();
  const navigate = useNavigate();
  const { sections, setCurrentSection, currentSection } = useGuide();

  useEffect(() => {
    if (section && sections.find(s => s.id === section)) {
      setCurrentSection(section);
    } else if (!section) {
      setCurrentSection('overview');
      navigate('/guide/overview', { replace: true });
    }
  }, [section, sections, setCurrentSection, navigate]);

  const currentSectionData = sections.find(s => s.id === currentSection);

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-50 backdrop-blur-sm bg-white/95">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => navigate('/')}
                className="flex items-center space-x-2 text-slate-600 hover:text-slate-900 transition-colors"
              >
                <ChevronLeft className="w-5 h-5" />
                <span>Back to Home</span>
              </button>
              <div className="h-6 w-px bg-slate-300" />
              <h1 className="text-xl font-semibold text-slate-900">ConversaLab Guide</h1>
            </div>
            <div className="flex items-center space-x-4">
              <ProgressBar />
              {currentSectionData && (
                <div className="flex items-center space-x-2">
                  {currentSectionData.completed && (
                    <CheckCircle className="w-5 h-5 text-green-500" />
                  )}
                  <span className="text-sm font-medium text-slate-600">
                    {currentSectionData.title}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto flex">
        {/* Sidebar */}
        <Sidebar />
        
        {/* Main Content */}
        <main className="flex-1 min-h-screen">
          <SectionContent />
        </main>
      </div>
    </div>
  );
}
