import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { toast } from "@/hooks/use-toast";
import { motion } from "framer-motion";
import { ArrowUp, ArrowDown } from "lucide-react";
import axios from "axios";

const BASELINE_2023 = 264.4;

const initial = {
  population: 71.7,         // million
  gdp: 11.24,               // trillion ฿
  primary_energy_consumption: 1390.81, // TWh
  oil_co2: 104.34,          // Mt
  coal_co2: 59.33,          // Mt
  cement_co2: 9.2,          // Mt (default, update as needed)
  total_ghg: 416.85,        // Mt CO₂e
  co2_including_luc: 297.37,// Mt
  temperature_change_from_ghg: 0.015, // °C
};

const Demo = () => {
  const [params, setParams] = useState(initial);
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState<number | null>(null);

  // For percent change calculation
  const percentChange =
    prediction !== null
      ? (((prediction - BASELINE_2023) / BASELINE_2023) * 100).toFixed(1)
      : null;

  const isIncrease = prediction !== null ? prediction > BASELINE_2023 : false;

  const handleChange = (key: keyof typeof initial, value: number) => {
    setParams((prev) => ({ ...prev, [key]: value }));
  };

  const handlePredict = async () => {
    setIsLoading(true);
    setPrediction(null);
    try {
      // Prepare payload
      const payload = {
        population: params.population * 1_000_000,
        gdp: params.gdp * 1_000_000_000_000,
        primary_energy_consumption: params.primary_energy_consumption,
        oil_co2: params.oil_co2,
        coal_co2: params.coal_co2,
        cement_co2: params.cement_co2,
        total_ghg: params.total_ghg,
        co2_including_luc: params.co2_including_luc,
        temperature_change_from_ghg: params.temperature_change_from_ghg,
      };

      const response = await axios.post("http://localhost:8000/predict", payload, {
        headers: { "Content-Type": "application/json" },
      });

      if (response.data && typeof response.data.prediction === "number") {
        setPrediction(Number(response.data.prediction));
        toast({
          title: "Prediction Success",
          description: `Predicted CO₂ emissions: ${response.data.prediction.toFixed(1)} Mt`,
        });
      } else {
        throw new Error("API format error");
      }
    } catch (err: any) {
      toast({
        title: "Prediction Failed",
        description: err?.message || "An error occurred.",
        variant: "destructive",
      });
    }
    setIsLoading(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-tr from-blue-50 via-violet-50 to-green-50 pt-16 px-2 pb-24 animate-fade-in">
      <motion.div
        initial={{ opacity: 0, translateY: 24 }}
        animate={{ opacity: 1, translateY: 0 }}
        transition={{ duration: 0.7 }}
        className="max-w-3xl mx-auto"
      >
        <Card className="shadow-md border-slate-200">
          <CardHeader>
            <CardTitle className="text-slate-800 text-3xl mb-1 tracking-tight">Thailand CO₂ Emissions Prediction</CardTitle>
            <CardDescription className="mb-2">
              Adjust parameters below and run the model to forecast total annual CO₂ emissions.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form
              className="space-y-6"
              onSubmit={(e) => {
                e.preventDefault();
                handlePredict();
              }}
            >
              <div className="grid gap-6">
                {/* Population */}
                <div>
                  <div className="flex justify-between mb-1">
                    <Label htmlFor="population">Population (Million)</Label>
                    <span className="font-mono tabular-nums text-primary">{params.population.toFixed(1)}</span>
                  </div>
                  <Slider
                    id="population"
                    min={65}
                    max={80}
                    step={0.1}
                    value={[params.population]}
                    onValueChange={(v) => handleChange("population", v[0])}
                  />
                  <div className="flex justify-between text-xs mt-0.5 text-muted-foreground"><span>65M</span><span>80M</span></div>
                </div>
                {/* GDP */}
                <div>
                  <div className="flex justify-between mb-1">
                    <Label htmlFor="gdp">GDP (Trillion ฿)</Label>
                    <span className="font-mono tabular-nums text-primary">{params.gdp.toFixed(2)}</span>
                  </div>
                  <Slider
                    id="gdp"
                    min={10}
                    max={20}
                    step={0.01}
                    value={[params.gdp]}
                    onValueChange={(v) => handleChange("gdp", v[0])}
                  />
                  <div className="flex justify-between text-xs mt-0.5 text-muted-foreground"><span>10T</span><span>20T</span></div>
                </div>
                {/* Primary Energy */}
                <div>
                  <div className="flex justify-between mb-1">
                    <Label htmlFor="primary_energy_consumption">Primary Energy (TWh)</Label>
                    <span className="font-mono tabular-nums text-primary">{params.primary_energy_consumption.toFixed(1)}</span>
                  </div>
                  <Slider
                    id="primary_energy_consumption"
                    min={1200}
                    max={1600}
                    step={0.1}
                    value={[params.primary_energy_consumption]}
                    onValueChange={(v) => handleChange("primary_energy_consumption", v[0])}
                  />
                  <div className="flex justify-between text-xs mt-0.5 text-muted-foreground"><span>1200</span><span>1600</span></div>
                </div>
                {/* Oil CO₂ */}
                <div>
                  <div className="flex justify-between mb-1">
                    <Label htmlFor="oil_co2">Oil CO₂ (Mt)</Label>
                    <span className="font-mono tabular-nums text-primary">{params.oil_co2.toFixed(1)}</span>
                  </div>
                  <Slider
                    id="oil_co2"
                    min={80}
                    max={130}
                    step={0.1}
                    value={[params.oil_co2]}
                    onValueChange={(v) => handleChange("oil_co2", v[0])}
                  />
                  <div className="flex justify-between text-xs mt-0.5 text-muted-foreground"><span>80</span><span>130</span></div>
                </div>
                {/* Coal CO₂ */}
                <div>
                  <div className="flex justify-between mb-1">
                    <Label htmlFor="coal_co2">Coal CO₂ (Mt)</Label>
                    <span className="font-mono tabular-nums text-primary">{params.coal_co2.toFixed(1)}</span>
                  </div>
                  <Slider
                    id="coal_co2"
                    min={50}
                    max={90}
                    step={0.1}
                    value={[params.coal_co2]}
                    onValueChange={(v) => handleChange("coal_co2", v[0])}
                  />
                  <div className="flex justify-between text-xs mt-0.5 text-muted-foreground"><span>50</span><span>90</span></div>
                </div>
                {/* Cement CO₂ */}
                <div>
                  <div className="flex justify-between mb-1">
                    <Label htmlFor="cement_co2">Cement CO₂ (Mt)</Label>
                    <span className="font-mono tabular-nums text-primary">{params.cement_co2.toFixed(1)}</span>
                  </div>
                  <Slider
                    id="cement_co2"
                    min={4}
                    max={15}
                    step={0.1}
                    value={[params.cement_co2]}
                    onValueChange={(v) => handleChange("cement_co2", v[0])}
                  />
                  <div className="flex justify-between text-xs mt-0.5 text-muted-foreground"><span>4</span><span>15</span></div>
                </div>
                {/* Total GHG */}
                <div>
                  <div className="flex justify-between mb-1">
                    <Label htmlFor="total_ghg">Total GHG (Mt CO₂e)</Label>
                    <span className="font-mono tabular-nums text-primary">{params.total_ghg.toFixed(1)}</span>
                  </div>
                  <Slider
                    id="total_ghg"
                    min={400}
                    max={500}
                    step={0.1}
                    value={[params.total_ghg]}
                    onValueChange={(v) => handleChange("total_ghg", v[0])}
                  />
                  <div className="flex justify-between text-xs mt-0.5 text-muted-foreground"><span>400</span><span>500</span></div>
                </div>
                {/* CO₂ incl. LUC */}
                <div>
                  <div className="flex justify-between mb-1">
                    <Label htmlFor="co2_including_luc">CO₂ incl. land-use change (Mt)</Label>
                    <span className="font-mono tabular-nums text-primary">{params.co2_including_luc.toFixed(1)}</span>
                  </div>
                  <Slider
                    id="co2_including_luc"
                    min={250}
                    max={350}
                    step={0.1}
                    value={[params.co2_including_luc]}
                    onValueChange={(v) => handleChange("co2_including_luc", v[0])}
                  />
                  <div className="flex justify-between text-xs mt-0.5 text-muted-foreground"><span>250</span><span>350</span></div>
                </div>
                {/* Temp Change */}
                <div>
                  <div className="flex justify-between mb-1">
                    <Label htmlFor="temperature_change_from_ghg">Temp Change from GHG (°C)</Label>
                    <span className="font-mono tabular-nums text-primary">{params.temperature_change_from_ghg.toFixed(3)}</span>
                  </div>
                  <Slider
                    id="temperature_change_from_ghg"
                    min={0}
                    max={2}
                    step={0.001}
                    value={[params.temperature_change_from_ghg]}
                    onValueChange={(v) => handleChange("temperature_change_from_ghg", v[0])}
                  />
                  <div className="flex justify-between text-xs mt-0.5 text-muted-foreground"><span>0°C</span><span>2°C</span></div>
                </div>
              </div>
              <Button
                type="submit"
                className="w-full text-lg py-5 bg-blue-600 hover:bg-blue-700 transition-all"
                disabled={isLoading}
              >
                {isLoading ? (
                  <>
                    <svg className="animate-spin mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                    </svg>
                    Forecasting...
                  </>
                ) : (
                  <>Predict CO₂ Emissions</>
                )}
              </Button>
            </form>
            {/* Prediction Result Card */}
            {prediction !== null && (
              <motion.div
                initial={{ opacity: 0, scale: 0.97 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.4 }}
                className="mt-8"
              >
                <Card className="w-full bg-gradient-to-br from-blue-100 via-indigo-50 to-green-100 py-6 px-4 border-blue-100 rounded-lg shadow-lg animate-[fade-in_0.8s]">
                  <div className="text-center">
                    <div className="text-gray-500 mb-1 text-sm">Predicted Emissions (Current Configuration)</div>
                    <div className="text-5xl font-bold text-blue-800 mb-2 flex items-center justify-center">
                      {prediction.toFixed(1)}<span className="text-lg font-normal text-blue-600 ml-1">Mt CO₂</span>
                    </div>
                    <div className="flex justify-center items-center gap-2 text-sm mt-1">
                      <span className="text-gray-700">vs 2023 baseline:</span>
                      <span className={`font-medium flex items-center ${isIncrease ? "text-red-500" : "text-green-600"}`}>
                        {isIncrease ? (
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
                      <span className="text-muted-foreground ml-2">({BASELINE_2023} Mt in 2023)</span>
                    </div>
                  </div>
                </Card>
              </motion.div>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
};

export default Demo;
