import logo from './logo.svg';
import './App.css';
import OllamaGPUCalculator from './OllamaGPUCalculator';
import Analytics from './components/Analytics';

function App() {
  return (
    <div className="App">
      <Analytics />
      <OllamaGPUCalculator />
    </div>
  );
}

export default App;
