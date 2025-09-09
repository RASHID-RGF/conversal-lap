import { useGuide } from '@/react-app/contexts/GuideContext';

export default function ProgressBar() {
  const { progress, sections } = useGuide();
  const completedCount = sections.filter(s => s.completed).length;
  const totalCount = sections.length;

  return (
    <div className="flex items-center space-x-3">
      <div className="w-32 h-2 bg-slate-200 rounded-full overflow-hidden">
        <div 
          className="h-full bg-gradient-to-r from-green-500 to-emerald-500 transition-all duration-500 ease-out"
          style={{ width: `${progress}%` }}
        />
      </div>
      <span className="text-sm font-medium text-slate-600 min-w-0">
        {completedCount}/{totalCount}
      </span>
    </div>
  );
}
