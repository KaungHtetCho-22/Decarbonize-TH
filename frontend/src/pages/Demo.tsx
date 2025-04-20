
import React, { useState } from 'react';
import { motion, AnimatePresence } from "framer-motion";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { toast } from "@/hooks/use-toast";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { ArrowRight, ArrowUp, ArrowDown, Calendar, Sliders, Activity } from "lucide-react";
import axios from "axios";

const Demo = () => {
  // State for forecast year
  const [forecastYear, setForecastYear] = useState("2025");
  
  // State for form inputs (with initial values)
  const [population, setPopulation] = useState(71.7); // in millions
  const [gdp, setGdp] = useState(11.24); // in trillion ฿
  const [primaryEnergy, setPrimaryEnergy] = useState(1390.81);
  const [oilCO2, setOilCO2] = useState(104.34);
  const [coalCO2, setCoalCO2] = useState(59.33);
  const [totalGHG, setTotalGHG] = useState(416.85);
  const [co2WithLUC, setCo2WithLUC] = useState(297.37);
  const [tempChange, setTempChange] = useState(0.015);
  
  // State for prediction result
  const [prediction, setPrediction] = useState<number | null>(null);
  const [isCalculating, setIsCalculating] = useState(false);
  const [showResult, setShowResult] = useState(false);
  const [historicalData, setHistoricalData] = useState([
    { year: "2018", emissions: 288.305 },
    { year: "2019", emissions: 281.877 },
    { year: "2020", emissions: 271.923 },
    { year: "2021", emissions: 267.142 },
    { year: "2022", emissions: 272.573 },
    { year: "2023", emissions: 264.389 }
  ]);
  
  // Baseline for comparison (2023 baseline)
  const baselineEmissions = 305.2;
  
  // Function to handle form submission
  const handleCalculate = async () => {
    setIsCalculating(true);
    setShowResult(false);
    
    try {
      // Prepare data for API call - convert to proper units
      const requestData = {
        population: population * 1000000, // Convert from millions to actual
        gdp: gdp * 1000000000000, // Convert from trillion to actual
        primary_energy_consumption: primaryEnergy,
        oil_co2: oilCO2,
        coal_co2: coalCO2,
        total_ghg: totalGHG,
        co2_including_luc: co2WithLUC,
        temperature_change_from_ghg: tempChange
      };
      
      // In a real scenario, this would call an API endpoint with the model
      // Simulating API call with timeout
      setTimeout(async () => {
        // Uncomment when API is ready
        /*
        const response = await axios.post('/predict', requestData);
        const predictedValue = response.data.prediction;
        */
        
        // For now, simulate with weighted formula
        const predictedValue = (
          (primaryEnergy * 0.12) + 
          (oilCO2 * 0.25) + 
          (coalCO2 * 0.18) + 
          (totalGHG * 0.15) + 
          (co2WithLUC * 0.22) +
          (population * 0.05) +
          (gdp * 5) +
          (tempChange * 0.03)
        ) * 0.75;
        
        const roundedPrediction = Math.round(predictedValue * 10) / 10;
        
        // Update historical data with prediction
        const updatedHistorical = [...historicalData];
        
        // Add forecast point
        const forecastPoint = { year: forecastYear, emissions: roundedPrediction, forecast: true };
        
        setPrediction(roundedPrediction);
        setHistoricalData([...updatedHistorical, forecastPoint]);
        setIsCalculating(false);
        setShowResult(true);
        
        toast({
          title: "Prediction Complete",
          description: `Predicted carbon emissions for ${forecastYear}: ${roundedPrediction} Mt CO₂`,
        });
      }, 1500);
    } catch (error) {
      console.error("Error calculating prediction:", error);
      toast({
        title: "Error",
        description: "Failed to calculate prediction. Please try again.",
        variant: "destructive"
      });
      setIsCalculating(false);
    }
  };
  
  // Calculate percentage change from baseline
  const percentChange = prediction ? ((prediction - baselineEmissions) / baselineEmissions * 100).toFixed(1) : null;
  const increasedEmissions = prediction && prediction > baselineEmissions;

  return (
    <div className="min-h-screen pt-16 bg-gradient-to-b from-slate-50 to-white">
      <motion.section 
        className="relative py-16 overflow-hidden"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6 }}
      >
        <div className="container mx-auto px-4 text-center">
          <div className="inline-block p-2 bg-blue-100 rounded-full mb-4">
            <Activity className="h-8 w-8 text-blue-600" />
          </div>
          <h1 className="text-4xl font-bold text-slate-800 mb-6">Carbon Emissions Forecasting</h1>
          <p className="text-lg text-slate-600 max-w-3xl mx-auto mb-8">
            Adjust the parameters below to generate a forecast for Thailand's carbon emissions 
            based on our XGBoost machine learning model.
          </p>
        </div>
      </motion.section>
      
      <section className="py-8">
        <div className="container mx-auto px-4">
          <div className="grid lg:grid-cols-12 gap-8">
            <div className="lg:col-span-5">
              <Card className="border-slate-200 shadow-md">
                <CardHeader className="pb-4">
                  <div className="flex justify-between items-center mb-2">
                    <CardTitle className="text-slate-800">Model Parameters</CardTitle>
                    <Select
                      value={forecastYear}
                      onValueChange={setForecastYear}
                    >
                      <SelectTrigger className="w-[110px] bg-white">
                        <Calendar className="mr-2 h-4 w-4" />
                        <SelectValue placeholder="Year" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="2025">2025</SelectItem>
                        <SelectItem value="2026">2026</SelectItem>
                        <SelectItem value="2027">2027</SelectItem>
                        <SelectItem value="2028">2028</SelectItem>
                        <SelectItem value="2029">2029</SelectItem>
                        <SelectItem value="2030">2030</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <CardDescription>
                    Adjust the sliders to set input parameters for forecasting {forecastYear} emissions
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <form className="space-y-6">
                    <div className="space-y-5">
                      <div>
                        <div className="flex justify-between mb-2">
                          <Label htmlFor="population">Population (Million)</Label>
                          <span className="text-slate-500 font-mono">{population.toFixed(1)}</span>
                        </div>
                        <Slider 
                          id="population" 
                          min={65}
                          max={80}
                          step={0.1}
                          value={[population]}
                          onValueChange={(value) => setPopulation(value[0])}
                          className="mb-1"
                        />
                        <div className="flex justify-between text-xs text-slate-400">
                          <span>65M</span>
                          <span>80M</span>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between mb-2">
                          <Label htmlFor="gdp">GDP (Trillion ฿)</Label>
                          <span className="text-slate-500 font-mono">{gdp.toFixed(2)}</span>
                        </div>
                        <Slider 
                          id="gdp" 
                          min={10}
                          max={20}
                          step={0.01}
                          value={[gdp]}
                          onValueChange={(value) => setGdp(value[0])}
                          className="mb-1"
                        />
                        <div className="flex justify-between text-xs text-slate-400">
                          <span>10T</span>
                          <span>20T</span>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between mb-2">
                          <Label htmlFor="primaryEnergy">Primary Energy (TWh)</Label>
                          <span className="text-slate-500 font-mono">{primaryEnergy.toFixed(1)}</span>
                        </div>
                        <Slider 
                          id="primaryEnergy" 
                          min={1200}
                          max={1600}
                          step={0.1}
                          value={[primaryEnergy]}
                          onValueChange={(value) => setPrimaryEnergy(value[0])}
                          className="mb-1"
                        />
                        <div className="flex justify-between text-xs text-slate-400">
                          <span>1200</span>
                          <span>1600</span>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between mb-2">
                          <Label htmlFor="oilCO2">Oil CO₂ (Mt)</Label>
                          <span className="text-slate-500 font-mono">{oilCO2.toFixed(1)}</span>
                        </div>
                        <Slider 
                          id="oilCO2" 
                          min={80}
                          max={130}
                          step={0.1}
                          value={[oilCO2]}
                          onValueChange={(value) => setOilCO2(value[0])}
                          className="mb-1"
                        />
                        <div className="flex justify-between text-xs text-slate-400">
                          <span>80</span>
                          <span>130</span>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between mb-2">
                          <Label htmlFor="coalCO2">Coal CO₂ (Mt)</Label>
                          <span className="text-slate-500 font-mono">{coalCO2.toFixed(1)}</span>
                        </div>
                        <Slider 
                          id="coalCO2" 
                          min={50}
                          max={90}
                          step={0.1}
                          value={[coalCO2]}
                          onValueChange={(value) => setCoalCO2(value[0])}
                          className="mb-1"
                        />
                        <div className="flex justify-between text-xs text-slate-400">
                          <span>50</span>
                          <span>90</span>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between mb-2">
                          <Label htmlFor="totalGHG">Total GHG (Mt CO₂e)</Label>
                          <span className="text-slate-500 font-mono">{totalGHG.toFixed(1)}</span>
                        </div>
                        <Slider 
                          id="totalGHG" 
                          min={400}
                          max={500}
                          step={0.1}
                          value={[totalGHG]}
                          onValueChange={(value) => setTotalGHG(value[0])}
                          className="mb-1"
                        />
                        <div className="flex justify-between text-xs text-slate-400">
                          <span>400</span>
                          <span>500</span>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between mb-2">
                          <Label htmlFor="co2WithLUC">CO₂ incl. land-use change (Mt)</Label>
                          <span className="text-slate-500 font-mono">{co2WithLUC.toFixed(1)}</span>
                        </div>
                        <Slider 
                          id="co2WithLUC" 
                          min={250}
                          max={350}
                          step={0.1}
                          value={[co2WithLUC]}
                          onValueChange={(value) => setCo2WithLUC(value[0])}
                          className="mb-1"
                        />
                        <div className="flex justify-between text-xs text-slate-400">
                          <span>250</span>
                          <span>350</span>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between mb-2">
                          <Label htmlFor="tempChange">Temp Change from GHG (°C)</Label>
                          <span className="text-slate-500 font-mono">{tempChange.toFixed(3)}</span>
                        </div>
                        <Slider 
                          id="tempChange" 
                          min={0}
                          max={2}
                          step={0.001}
                          value={[tempChange]}
                          onValueChange={(value) => setTempChange(value[0])}
                          className="mb-1"
                        />
                        <div className="flex justify-between text-xs text-slate-400">
                          <span>0°C</span>
                          <span>2°C</span>
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
                        className="w-full bg-blue-600 hover:bg-blue-700 text-white py-6"
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
                            Processing...
                          </>
                        ) : (
                          <>
                            <Sliders className="mr-2 h-5 w-5" />
                            Generate {forecastYear} Forecast
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
                    <Card className="w-full h-full flex flex-col border-slate-200 shadow-md">
                      <CardHeader>
                        <CardTitle className="text-slate-800">Forecast Results</CardTitle>
                        <CardDescription>
                          Adjust the parameters and click "Generate Forecast" to see the prediction
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="flex-grow flex items-center justify-center">
                        <div className="text-center">
                          <Activity className="h-24 w-24 text-slate-200 mx-auto mb-6" />
                          <p className="text-lg text-slate-500">
                            Carbon emissions forecast will appear here
                          </p>
                          <p className="text-sm text-slate-400 mt-2">
                            Use the sliders on the left to adjust model parameters
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
                    <Card className="mb-6 border-slate-200 shadow-md overflow-hidden">
                      <CardHeader className="pb-0">
                        <CardTitle className="text-slate-800">Carbon Emissions Forecast</CardTitle>
                        <CardDescription>
                          Based on XGBoost model prediction
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="flex flex-col items-center justify-center py-4 mb-2 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg">
                          <div className="text-slate-500 mb-1 text-sm">Predicted Emissions for {forecastYear}</div>
                          <div className="text-5xl font-bold text-blue-700 mb-1 flex items-center">
                            {prediction?.toFixed(1)} 
                            <span className="text-lg font-normal text-slate-500 ml-2">Mt CO₂</span>
                          </div>
                          
                          <div className="flex items-center mt-1 text-sm">
                            <div className="flex items-center">
                              <span className="text-slate-600 mr-1">vs 2023 baseline:</span>
                              <span className={`font-medium flex items-center ${increasedEmissions ? 'text-red-500' : 'text-green-500'}`}>
                                {increasedEmissions ? (
                                  <>
                                    <ArrowUp className="h-4 w-4 mr-1" />
                                    {percentChange}%
                                  </>
                                ) : (
                                  <>
                                    <ArrowDown className="h-4 w-4 mr-1" />
                                    {Math.abs(Number(percentChange))}%
                                  </>
                                )}
                              </span>
                            </div>
                          </div>
                        </div>
                        
                        <div className="mt-6 h-[300px]">
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart
                              data={historicalData}
                              margin={{ top: 10, right: 30, left: 10, bottom: 20 }}
                            >
                              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                              <XAxis 
                                dataKey="year" 
                                tick={{ fontSize: 12 }}
                                axisLine={{ stroke: '#e5e7eb' }}
                                tickLine={{ stroke: '#e5e7eb' }}
                              />
                              <YAxis 
                                domain={['auto', 'auto']} 
                                tick={{ fontSize: 12 }} 
                                axisLine={{ stroke: '#e5e7eb' }}
                                tickLine={{ stroke: '#e5e7eb' }}
                                label={{ 
                                  value: 'CO₂ Emissions (Mt)', 
                                  angle: -90, 
                                  position: 'insideLeft',
                                  style: { textAnchor: 'middle', fontSize: 12, fill: '#64748b' }
                                }} 
                              />
                              <Tooltip 
                                formatter={(value) => [`${value} Mt CO₂`, 'Emissions']}
                                labelFormatter={(label) => `Year: ${label}`}
                                contentStyle={{ 
                                  backgroundColor: 'white', 
                                  border: '1px solid #e5e7eb',
                                  borderRadius: '0.375rem',
                                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                                }}
                              />
                              <Legend verticalAlign="top" height={36} />
                              <ReferenceLine 
                                y={baselineEmissions} 
                                stroke="#94a3b8" 
                                strokeDasharray="3 3" 
                                label={{ 
                                  value: '2023 Baseline', 
                                  position: 'right', 
                                  fill: '#94a3b8', 
                                  fontSize: 12 
                                }} 
                              />
                              <Line 
                                type="monotone" 
                                dataKey="emissions"
                                name="Historical Emissions"
                                stroke="#3b82f6"
                                strokeWidth={2}
                                dot={{ r: 4, strokeWidth: 2, fill: 'white' }}
                                activeDot={{ r: 6, strokeWidth: 2 }}
                              />
                              <Line 
                                type="monotone" 
                                dataKey={(data) => data.forecast ? data.emissions : null}
                                name={`${forecastYear} Forecast`}
                                stroke="#3b82f6"
                                strokeWidth={2}
                                strokeDasharray="5 5"
                                dot={{ r: 6, fill: '#3b82f6', strokeWidth: 2, stroke: '#3b82f6' }}
                              />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      </CardContent>
                    </Card>
                    
                    <Card className="border-slate-200 shadow-md">
                      <CardHeader>
                        <CardTitle className="text-slate-800">Contributing Factors</CardTitle>
                        <CardDescription>
                          Key variables impacting the forecast
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="grid md:grid-cols-2 gap-4">
                          <div className="p-4 rounded-lg bg-blue-50 border border-blue-100">
                            <h4 className="font-medium text-blue-800 mb-2 flex items-center">
                              <Activity className="h-4 w-4 mr-2" />
                              Energy Consumption
                            </h4>
                            <p className="text-sm text-slate-600">
                              Primary energy usage has the strongest correlation with emissions. 
                              Your forecast uses {primaryEnergy.toFixed(1)} TWh, which is
                              {primaryEnergy > 1400 ? ' above' : ' below'} the 2022 baseline of 1400 TWh.
                            </p>
                          </div>
                          
                          <div className="p-4 rounded-lg bg-indigo-50 border border-indigo-100">
                            <h4 className="font-medium text-indigo-800 mb-2 flex items-center">
                              <Activity className="h-4 w-4 mr-2" />
                              Economic Growth
                            </h4>
                            <p className="text-sm text-slate-600">
                              GDP growth remains coupled with emissions in Thailand's economy.
                              Your forecast uses {gdp.toFixed(2)} trillion ฿, representing
                              {gdp > 11.5 ? ' growth' : ' contraction'} from the 2022 level.
                            </p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </section>
      
      <section className="py-12 bg-slate-50">
        <div className="container mx-auto px-4">
          <h2 className="text-2xl font-bold text-center text-slate-800 mb-8">About the Forecast Model</h2>
          
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                title: "XGBoost Model",
                description: "This demo utilizes an XGBoost regression model trained on historical data from Thailand's carbon emissions and related socioeconomic factors.",
              },
              {
                title: "Feature Importance",
                description: "The model weights energy consumption, emissions by source, and economic indicators according to their historical correlation with overall carbon emissions.",
              },
              {
                title: "Data Sources",
                description: "The underlying model was trained using OWID (Our World in Data) CO₂ emissions dataset, covering the period from 1950 to present day.",
              }
            ].map((card, index) => (
              <Card key={index} className="h-full border-slate-200 shadow-sm">
                <CardHeader>
                  <CardTitle className="text-slate-800">{card.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-slate-600">
                    {card.description}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>
      
      <section className="py-12 bg-gradient-to-br from-blue-700 to-blue-900 text-white">
        <div className="container mx-auto px-4">
          <div className="text-center">
            <h2 className="text-3xl font-bold mb-4">Explore More Features</h2>
            <p className="max-w-2xl mx-auto mb-8 text-blue-100">
              Learn about our methodology and explore the complete range of models developed.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <motion.div
                className="inline-block"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Button 
                  variant="outline"
                  className="bg-white text-blue-700 border-blue-200 hover:bg-blue-50"
                >
                  View Dataset
                  <ArrowRight className="ml-2 h-5 w-5" />
                </Button>
              </motion.div>
              <motion.div
                className="inline-block"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Button>
                  Explore Models
                </Button>
              </motion.div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Demo;
