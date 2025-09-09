import { BrowserRouter as Router, Routes, Route } from "react-router";
import HomePage from "@/react-app/pages/Home";
import GuidePage from "@/react-app/pages/Guide";
import { GuideProvider } from "@/react-app/contexts/GuideContext";

export default function App() {
  return (
    <GuideProvider>
      <Router>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/guide/:section?" element={<GuidePage />} />
        </Routes>
      </Router>
    </GuideProvider>
  );
}
