import { useNavigate } from 'react-router';
import { Brain, Rocket, Shield, Zap, Users, Cloud, Code, BarChart3 } from 'lucide-react';
import { useEffect, useState } from 'react';

export default function Home() {
  const navigate = useNavigate();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const features = [
    {
      icon: Brain,
      title: "Core Components",
      description: "Learn about LLMs, tokenizers, and inference systems",
      color: "from-purple-500 to-pink-500"
    },
    {
      icon: BarChart3,
      title: "Data Management",
      description: "Collect, preprocess, and manage training datasets safely",
      color: "from-blue-500 to-cyan-500"
    },
    {
      icon: Code,
      title: "Open Source Tools",
      description: "Explore Hugging Face, LLaMA, Mistral, and more",
      color: "from-green-500 to-emerald-500"
    },
    {
      icon: Cloud,
      title: "Infrastructure",
      description: "Hardware requirements and cost-effective cloud options",
      color: "from-orange-500 to-red-500"
    },
    {
      icon: Zap,
      title: "Fine-tuning",
      description: "Customize models for specific use cases",
      color: "from-yellow-500 to-orange-500"
    },
    {
      icon: Shield,
      title: "Safety & Ethics",
      description: "Implement moderation and alignment techniques",
      color: "from-indigo-500 to-purple-500"
    },
    {
      icon: Rocket,
      title: "API & Integration",
      description: "Build user-friendly APIs and app integrations",
      color: "from-pink-500 to-rose-500"
    },
    {
      icon: Users,
      title: "Scaling & Maintenance",
      description: "Best practices for production deployment",
      color: "from-teal-500 to-green-500"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Animated background */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -inset-10 opacity-20">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl animate-pulse"></div>
          <div className="absolute top-3/4 right-1/4 w-96 h-96 bg-cyan-500 rounded-full mix-blend-multiply filter blur-xl animate-pulse animation-delay-2000"></div>
          <div className="absolute bottom-1/4 left-1/2 w-96 h-96 bg-pink-500 rounded-full mix-blend-multiply filter blur-xl animate-pulse animation-delay-4000"></div>
        </div>
      </div>

      <div className="relative">
        {/* Header */}
        <header className="px-6 py-8">
          <nav className="flex items-center justify-between max-w-7xl mx-auto">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <h1 className="text-2xl font-bold text-white">ConversaLab</h1>
            </div>
            <button
              onClick={() => navigate('/guide')}
              className="px-6 py-3 bg-white/10 hover:bg-white/20 text-white rounded-lg backdrop-blur-sm border border-white/20 hover:border-white/30 transition-all duration-300"
            >
              Get Started
            </button>
          </nav>
        </header>

        {/* Hero Section */}
        <section className="px-6 py-20">
          <div className={`max-w-4xl mx-auto text-center transition-all duration-1000 ${mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
            <h2 className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight">
              Build Advanced
              <span className="block bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 bg-clip-text text-transparent animated-gradient">
                Conversational AI
              </span>
            </h2>
            <p className="text-xl md:text-2xl text-slate-300 mb-12 max-w-3xl mx-auto leading-relaxed">
              Your comprehensive guide to designing, building, and deploying AI systems like ChatGPT. 
              From core components to production deployment.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <button
                onClick={() => navigate('/guide')}
                className="px-8 py-4 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-xl font-semibold text-lg hover:shadow-2xl hover:shadow-purple-500/25 transform hover:scale-105 transition-all duration-300"
              >
                Start Learning
              </button>
              <button
                onClick={() => navigate('/guide/overview')}
                className="px-8 py-4 bg-white/10 text-white rounded-xl font-semibold text-lg backdrop-blur-sm border border-white/20 hover:bg-white/20 transition-all duration-300"
              >
                View Overview
              </button>
            </div>
          </div>
        </section>

        {/* Features Grid */}
        <section className="px-6 py-20">
          <div className="max-w-7xl mx-auto">
            <h3 className="text-3xl md:text-4xl font-bold text-white text-center mb-16">
              Everything You Need to Know
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {features.map((feature, index) => {
                const Icon = feature.icon;
                return (
                  <div
                    key={index}
                    className={`group p-6 rounded-2xl glass hover-lift cursor-pointer transition-all duration-500 ${mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}
                    style={{ transitionDelay: `${index * 100}ms` }}
                    onClick={() => navigate('/guide')}
                  >
                    <div className={`w-12 h-12 rounded-xl bg-gradient-to-r ${feature.color} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300`}>
                      <Icon className="w-6 h-6 text-white" />
                    </div>
                    <h4 className="text-xl font-semibold text-white mb-2">
                      {feature.title}
                    </h4>
                    <p className="text-slate-300 leading-relaxed">
                      {feature.description}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="px-6 py-20">
          <div className="max-w-4xl mx-auto text-center">
            <div className="p-12 rounded-3xl glass">
              <h3 className="text-3xl md:text-4xl font-bold text-white mb-6">
                Ready to Build the Future?
              </h3>
              <p className="text-xl text-slate-300 mb-8 max-w-2xl mx-auto">
                Join thousands of developers learning to build sophisticated AI systems. 
                Start your journey today with our comprehensive step-by-step guide.
              </p>
              <button
                onClick={() => navigate('/guide')}
                className="px-10 py-5 bg-gradient-to-r from-cyan-500 to-blue-500 text-white rounded-xl font-semibold text-lg hover:shadow-2xl hover:shadow-cyan-500/25 transform hover:scale-105 transition-all duration-300"
              >
                Begin Your Journey
              </button>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
