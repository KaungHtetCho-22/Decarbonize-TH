import React, { useState } from 'react';
import { motion, AnimatePresence } from "framer-motion";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { toast } from "@/components/ui/use-toast";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts";
import { Play, BarChart3, Activity, ArrowRight } from "lucide-react";

const Demo = () => {
  // State for form inputs
  const [energy, setEnergy] = useState(200);
  const [gdp, setGdp] = useState(15);
  const [population, setPopulation] = useState(70);
  const [industry, setIndustry] = useState(110);
  const [transport, setTransport] = useState(95);
  const [agriculture, setAgriculture] = useState(80);
  const [temperature, setTemperature] = useState(28);
  
  // State for prediction result
  const [prediction, setPrediction] = useState<number | null>(null);
  const [isCalculating, setIsCalculating] = useState(false);
  const [showResult, setShowResult] = useState(false);
  const [historicalData, setHistoricalData] = useState([]);
  const [singleFactorData, setSingleFactorData] = useState([]);
  const [selectedFactor, setSelectedFactor] = useState("energy");
  
  // Map for impact descriptions
  const impactDescriptions = {
    "energy": "Energy consumption has the strongest direct impact on emissions. Every 1% increase in energy consumption typically results in a 0.8-0.9% increase in carbon emissions.",
    "gdp": "Economic growth has a significant but complex relationship with emissions. In Thailand, a 1% increase in GDP is associated with approximately a 0.6-0.7% increase in emissions, though this relationship is weakening as the economy becomes more service-oriented.",
    "population": "Population growth has a moderate effect, with a 1% increase typically leading to a 0.5% increase in emissions, reflecting per capita consumption patterns.",
    "industry": "Industrial output correlates strongly with emissions. A 1% increase in industrial activity typically leads to a 0.7-0.8% increase in carbon emissions.",
    "transport": "Transportation activity shows a direct relationship with emissions. A 1% increase in transport activity results in approximately a 0.7% increase in emissions.",
    "agriculture": "Agricultural production has a moderate impact, with 1% increases typically leading to 0.3-0.4% increases in emissions, primarily through land use changes and methane production."
  };
  
  // Function to handle form submission
  const handleCalculate = () => {
    setIsCalculating(true);
    setShowResult(false);
    
    // Simulated model prediction with a weighted formula
    // In a real scenario, this would call an API endpoint with the model
    setTimeout(() => {
      // Simple model simulation based on feature importance
      const predicted = (
        (energy * 0.278) + 
        (transport * 0.185) + 
        (industry * 0.162) + 
        (gdp * 12) + 
        (population * 0.079) +
        (agriculture * 0.018) +
        (temperature * 0.012)
      ) * 0.75;
      
      const roundedPrediction = Math.round(predicted * 10) / 10;
      
      // Generate historical data
      const historical = [
        { year: 2017, emissions: 274.6 },
        { year: 2018, emissions: 283.4 },
        { year: 2019, emissions: 291.7 },
        { year: 2020, emissions: 267.8 },
        { year: 2021, emissions: 281.9 },
        { year: 2022, emissions: roundedPrediction }
      ];
      
      // Generate single factor analysis data
      const generateFactorData = (factor) => {
        const baseValue = {
          energy: energy,
          gdp: gdp,
          population: population,
          industry: industry,
          transport: transport,
          agriculture: agriculture
        }[factor];
        
        const factorWeight = {
          energy: 0.278,
          gdp: 0.12,
          population: 0.079,
          industry: 0.162,
          transport: 0.185,
          agriculture: 0.018
        }[factor];
        
        return [
          { value: baseValue * 0.8, emissions: prediction * (1 - (factorWeight * 0.2)) },
          { value: baseValue * 0.9, emissions: prediction * (1 - (factorWeight * 0.1)) },
          { value: baseValue, emissions: prediction },
          { value: baseValue * 1.1, emissions: prediction * (1 + (factorWeight * 0.1)) },
          { value: baseValue * 1.2, emissions: prediction * (1 + (factorWeight * 0.2)) }
        ];
      };
      
      setPrediction(roundedPrediction);
      setHistoricalData(historical);
      setSingleFactorData(generateFactorData(selectedFactor));
      setIsCalculating(false);
      setShowResult(true);
      
      toast({
        title: "Prediction Complete",
        description: `Predicted carbon emissions: ${roundedPrediction} MT CO₂`,
      });
    }, 1500);
  };
  
  // Function to update single factor data when the selected factor changes
  const handleFactorChange = (factor) => {
    setSelectedFactor(factor);
    
    if (prediction) {
      const baseValue = {
        energy: energy,
        gdp: gdp,
        population: population,
        industry: industry,
        transport: transport,
        agriculture: agriculture
      }[factor];
      
      const factorWeight = {
        energy: 0.278,
        gdp: 0.12,
        population: 0.079,
        industry: 0.162,
        transport: 0.185,
        agriculture: 0.018
      }[factor];
      
      const factorData = [
        { value: baseValue * 0.8, emissions: prediction * (1 - (factorWeight * 0.2)) },
        { value: baseValue * 0.9, emissions: prediction * (1 - (factorWeight * 0.1)) },
        { value: baseValue, emissions: prediction },
        { value: baseValue * 1.1, emissions: prediction * (1 + (factorWeight * 0.1)) },
        { value: baseValue * 1.2, emissions: prediction * (1 + (factorWeight * 0.2)) }
      ];
      
      setSingleFactorData(factorData);
    }
  };
  
  const factorLabels = {
    "energy": "Energy Consumption (TWh)",
    "gdp": "GDP (Trillion ฿)",
    "population": "Population (Million)",
    "industry": "Industry Index",
    "transport": "Transport Activity (Billion km)",
    "agriculture": "Agricultural Production (Index)"
  };

  return (
    <div className="min-h-screen pt-16">
      <motion.section 
        className="relative py-16 bg-gradient-to-b from-green-50 to-white overflow-hidden"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6 }}
      >
        <div className="section-container text-center">
          <div className="inline-block p-2 bg-green-100 rounded-full mb-4">
            <Play className="h-8 w-8 text-green-600" />
          </div>
          <h1 className="text-4xl font-bold text-green-800 mb-6">Interactive Demo</h1>
          <p className="text-lg text-gray-700 max-w-3xl mx-auto mb-8">
            Experience our carbon emissions prediction model in action. Adjust the input values below 
            to see how different factors affect Thailand's carbon emissions.
          </p>
        </div>
        
        <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
          {[...Array(15)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute bg-green-300 opacity-20 rounded-full"
              style={{
                height: 5 + Math.random() * 20,
                width: 5 + Math.random() * 20,
                top: Math.random() * 100 + '%',
                left: Math.random() * 100 + '%',
              }}
              animate={{
                scale: [1, 1.5, 1],
                opacity: [0.2, 0.3, 0.2]
              }}
              transition={{
                duration: 3 + Math.random() * 2,
                repeat: Infinity,
                delay: Math.random() * 5
              }}
            />
          ))}
        </div>
      </motion.section>
      
      <section className="py-12">
        <div className="section-container">
          <div className="grid lg:grid-cols-12 gap-8">
            <div className="lg:col-span-5">
              <Card>
                <CardHeader>
                  <CardTitle className="text-green-700">Input Parameters</CardTitle>
                  <CardDescription>
                    Adjust the sliders to set input parameters for the prediction model
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <form className="space-y-6">
                    <div className="space-y-4">
                      <div>
                        <div className="flex justify-between mb-2">
                          <Label htmlFor="energy">Energy Consumption (TWh)</Label>
                          <span className="text-gray-500">{energy}</span>
                        </div>
                        <Slider 
                          id="energy" 
                          min={150}
                          max={250}
                          step={1}
                          value={[energy]}
                          onValueChange={(value) => setEnergy(value[0])}
                          className="mb-2"
                        />
                        <div className="flex justify-between text-xs text-gray-500">
                          <span>150</span>
                          <span>250</span>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between mb-2">
                          <Label htmlFor="gdp">GDP (Trillion ฿)</Label>
                          <span className="text-gray-500">{gdp}</span>
                        </div>
                        <Slider 
                          id="gdp" 
                          min={12}
                          max={18}
                          step={0.1}
                          value={[gdp]}
                          onValueChange={(value) => setGdp(value[0])}
                          className="mb-2"
                        />
                        <div className="flex justify-between text-xs text-gray-500">
                          <span>12</span>
                          <span>18</span>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between mb-2">
                          <Label htmlFor="population">Population (Million)</Label>
                          <span className="text-gray-500">{population}</span>
                        </div>
                        <Slider 
                          id="population" 
                          min={65}
                          max={75}
                          step={0.1}
                          value={[population]}
                          onValueChange={(value) => setPopulation(value[0])}
                          className="mb-2"
                        />
                        <div className="flex justify-between text-xs text-gray-500">
                          <span>65</span>
                          <span>75</span>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between mb-2">
                          <Label htmlFor="industry">Industrial Output (Index)</Label>
                          <span className="text-gray-500">{industry}</span>
                        </div>
                        <Slider 
                          id="industry" 
                          min={90}
                          max={130}
                          step={1}
                          value={[industry]}
                          onValueChange={(value) => setIndustry(value[0])}
                          className="mb-2"
                        />
                        <div className="flex justify-between text-xs text-gray-500">
                          <span>90</span>
                          <span>130</span>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between mb-2">
                          <Label htmlFor="transport">Transport Activity (Billion km)</Label>
                          <span className="text-gray-500">{transport}</span>
                        </div>
                        <Slider 
                          id="transport" 
                          min={75}
                          max={115}
                          step={1}
                          value={[transport]}
                          onValueChange={(value) => setTransport(value[0])}
                          className="mb-2"
                        />
                        <div className="flex justify-between text-xs text-gray-500">
                          <span>75</span>
                          <span>115</span>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between mb-2">
                          <Label htmlFor="agriculture">Agricultural Production (Index)</Label>
                          <span className="text-gray-500">{agriculture}</span>
                        </div>
                        <Slider 
                          id="agriculture" 
                          min={60}
                          max={100}
                          step={1}
                          value={[agriculture]}
                          onValueChange={(value) => setAgriculture(value[0])}
                          className="mb-2"
                        />
                        <div className="flex justify-between text-xs text-gray-500">
                          <span>60</span>
                          <span>100</span>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between mb-2">
                          <Label htmlFor="temperature">Average Temperature (°C)</Label>
                          <span className="text-gray-500">{temperature}</span>
                        </div>
                        <Slider 
                          id="temperature" 
                          min={26}
                          max={30}
                          step={0.1}
                          value={[temperature]}
                          onValueChange={(value) => setTemperature(value[0])}
                          className="mb-2"
                        />
                        <div className="flex justify-between text-xs text-gray-500">
                          <span>26</span>
                          <span>30</span>
                        </div>
                      </div>
                    </div>
                    
                    <motion.div
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      className="pt-4"
                    >
                      <Button
                        type="button"
                        className="w-full bg-green-600 hover:bg-green-700 text-white py-6"
                        size="lg"
                        onClick={handleCalculate}
                        disabled={isCalculating}
                      >
                        {isCalculating ? (
                          <>
                            <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            Calculating...
                          </>
                        ) : (
                          <>
                            <BarChart3 className="mr-2 h-5 w-5" />
                            Calculate Carbon Emissions
                          </>
                        )}
                      </Button>
                    </motion.div>
                  </form>
                </CardContent>
              </Card>
            </div>
            
            <div className="lg:col-span-7">
              <AnimatePresence>
                {!showResult && (
                  <motion.div
                    initial={{ opacity: 1 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex items-center justify-center h-full"
                  >
                    <Card className="w-full h-full flex flex-col">
                      <CardHeader>
                        <CardTitle className="text-green-700">Prediction Results</CardTitle>
                        <CardDescription>
                          Adjust the parameters and click "Calculate" to see the prediction
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="flex-grow flex items-center justify-center">
                        <div className="text-center">
                          <BarChart3 className="h-24 w-24 text-green-200 mx-auto mb-6" />
                          <p className="text-lg text-gray-500">
                            Your predicted carbon emissions will appear here
                          </p>
                          <p className="text-sm text-gray-400 mt-2">
                            Use the input sliders on the left to set parameters
                          </p>
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                )}
                
                {showResult && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    <Card className="mb-6">
                      <CardHeader>
                        <CardTitle className="text-green-700">Prediction Results</CardTitle>
                        <CardDescription>
                          Based on the provided parameters
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="flex flex-col items-center justify-center mb-6">
                          <div className="text-5xl font-bold text-green-700 mb-2">
                            {prediction} <span className="text-2xl font-normal text-gray-500">MT CO₂</span>
                          </div>
                          <p className="text-gray-600">
                            Predicted annual carbon emissions for Thailand
                          </p>
                        </div>
                        
                        <div className="h-[300px]">
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart
                              data={historicalData}
                              margin={{ top: 10, right: 30, left: 0, bottom: 10 }}
                            >
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="year" />
                              <YAxis domain={[260, 300]} label={{ value: 'CO₂ Emissions (MT)', angle: -90, position: 'insideLeft' }} />
                              <Tooltip 
                                formatter={(value) => (typeof value === 'number' ? value.toFixed(2) : value)}
                                labelFormatter={(label) => `Year: ${label}`}
                              />
                              <Legend verticalAlign="top" height={36} />
                              <Line 
                                type="monotone" 
                                dataKey="emissions" 
                                name="Carbon Emissions"
                                stroke="#22c55e" 
                                dot={{ stroke: '#15803d', strokeWidth: 2, r: 4 }} 
                                activeDot={{ r: 8 }} 
                                strokeWidth={2}
                              />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      </CardContent>
                    </Card>
                    
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-green-700">Factor Impact Analysis</CardTitle>
                        <CardDescription>
                          See how changes in individual factors affect emissions
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <Tabs defaultValue="energy" value={selectedFactor} onValueChange={handleFactorChange}>
                          <TabsList className="mb-6 overflow-x-auto flex w-full justify-start">
                            {Object.keys(factorLabels).map(factor => (
                              <TabsTrigger 
                                key={factor} 
                                value={factor}
                                className="flex-shrink-0 data-[state=active]:bg-green-600 data-[state=active]:text-white"
                              >
                                {factorLabels[factor].split(' ')[0]}
                              </TabsTrigger>
                            ))}
                          </TabsList>
                          
                          {Object.keys(factorLabels).map(factor => (
                            <TabsContent key={factor} value={factor}>
                              <div className="flex flex-col md:flex-row gap-6">
                                <div className="md:w-3/5">
                                  <h4 className="font-medium text-gray-800 mb-3">{factorLabels[factor]} Impact</h4>
                                  <div className="h-[250px]">
                                    <ResponsiveContainer width="100%" height="100%">
                                      <BarChart
                                        data={singleFactorData}
                                        margin={{ top: 5, right: 30, left: 20, bottom: 40 }}
                                      >
                                        <CartesianGrid strokeDasharray="3 3" />
                                        <XAxis 
                                          dataKey="value" 
                                          label={{ 
                                            value: factorLabels[factor], 
                                            position: 'bottom', 
                                            offset: 20 
                                          }} 
                                        />
                                        <YAxis 
                                          label={{ 
                                            value: 'CO₂ Emissions (MT)', 
                                            angle: -90, 
                                            position: 'insideLeft' 
                                          }}
                                        />
                                        <Tooltip 
                                          formatter={(value) => (typeof value === 'number' ? value.toFixed(2) : value)}
                                        />
                                        <Bar 
                                          dataKey="emissions" 
                                          name="Carbon Emissions" 
                                          fill="#22c55e" 
                                        />
                                      </BarChart>
                                    </ResponsiveContainer>
                                  </div>
                                </div>
                                
                                <div className="md:w-2/5 bg-green-50 p-4 rounded-md">
                                  <div className="flex items-center mb-3">
                                    <Activity className="h-5 w-5 text-green-600 mr-2" />
                                    <h4 className="font-medium text-green-800">Impact Analysis</h4>
                                  </div>
                                  <p className="text-gray-700 text-sm">
                                    {impactDescriptions[factor]}
                                  </p>
                                  <div className="mt-4 pt-4 border-t border-green-200">
                                    <h5 className="text-sm font-medium text-green-800 mb-2">Reduction Strategies:</h5>
                                    <ul className="text-sm text-gray-700 space-y-1 list-disc pl-5">
                                      {factor === "energy" && [
                                        "Invest in renewable energy sources",
                                        "Improve energy efficiency in buildings",
                                        "Implement smart grid technologies"
                                      ].map((item, i) => <li key={i}>{item}</li>)}
                                      
                                      {factor === "transport" && [
                                        "Expand public transportation networks",
                                        "Promote electric vehicle adoption",
                                        "Develop sustainable urban planning"
                                      ].map((item, i) => <li key={i}>{item}</li>)}
                                      
                                      {factor === "industry" && [
                                        "Adopt cleaner production technologies",
                                        "Implement carbon capture systems",
                                        "Incentivize energy efficiency measures"
                                      ].map((item, i) => <li key={i}>{item}</li>)}
                                      
                                      {factor === "gdp" && [
                                        "Transition to service-based economy",
                                        "Invest in green technology sectors",
                                        "Implement circular economy principles"
                                      ].map((item, i) => <li key={i}>{item}</li>)}
                                      
                                      {factor === "population" && [
                                        "Promote sustainable consumption patterns",
                                        "Reduce per capita carbon footprint",
                                        "Develop greener urban infrastructure"
                                      ].map((item, i) => <li key={i}>{item}</li>)}
                                      
                                      {factor === "agriculture" && [
                                        "Implement sustainable farming practices",
                                        "Reduce deforestation rates",
                                        "Promote regenerative agriculture techniques"
                                      ].map((item, i) => <li key={i}>{item}</li>)}
                                    </ul>
                                  </div>
                                </div>
                              </div>
                            </TabsContent>
                          ))}
                        </Tabs>
                      </CardContent>
                    </Card>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </section>
      
      <section className="py-12 bg-green-50">
        <div className="section-container">
          <h2 className="section-title">About the Prediction Model</h2>
          <p className="text-gray-700 mb-8">
            This demo utilizes a simplified version of our XGBoost machine learning model, which has been 
            trained on historical data from Thailand's carbon emissions and related factors.
          </p>
          
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                title: "How It Works",
                description: "The model analyzes relationships between economic, demographic, and environmental factors to predict carbon emissions. Each input parameter is weighted based on its historical impact on emission levels.",
              },
              {
                title: "Accuracy & Limitations",
                description: "While the full model achieves over 92% accuracy, this demo uses a simplified version. Actual predictions would require more detailed inputs and would account for complex interactions between factors.",
              },
              {
                title: "Data Sources",
                description: "The underlying model was trained using data from Thailand's Office of Natural Resources and Environmental Policy and Planning, the Electricity Generating Authority of Thailand, and other authoritative sources.",
              }
            ].map((card, index) => (
              <Card key={index} className="h-full">
                <CardHeader>
                  <CardTitle className="text-green-700">{card.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-700">
                    {card.description}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>
      
      <section className="py-12 bg-gradient-to-br from-green-700 to-green-900 text-white">
        <div className="section-container">
          <div className="text-center">
            <h2 className="text-3xl font-bold mb-4">Explore the Full Project</h2>
            <p className="max-w-2xl mx-auto mb-8 text-green-100">
              Learn more about our methodology, data sources, and the complete range of models we developed.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <motion.div
                className="inline-block"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <a 
                  href="/features" 
                  className="inline-flex items-center bg-white text-green-700 px-6 py-3 rounded-md font-medium hover:bg-green-50 transition-colors"
                >
                  Explore Features
                  <ArrowRight className="ml-2 h-5 w-5" />
                </a>
              </motion.div>
              <motion.div
                className="inline-block"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <a 
                  href="/" 
                  className="inline-flex items-center bg-green-600 border border-green-200 text-white px-6 py-3 rounded-md font-medium hover:bg-green-700 transition-colors"
                >
                  Back to Home
                </a>
              </motion.div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Demo;
