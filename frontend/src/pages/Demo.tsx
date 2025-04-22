
import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { toast } from "@/hooks/use-toast";
import { motion } from "framer-motion";
import { ArrowUp, ArrowDown } from "lucide-react";
import axios from "axios";
import { PredictionPayload, PredictionResponse } from "@/types/prediction";
<<<<<<< HEAD
import { BASELINE_2023, sliderConfigs } from "@/config/sliderConfig";

const Demo = () => {
  const [params, setParams] = useState(() => {
    return sliderConfigs.reduce((acc, config) => {
      return { ...acc, [config.id]: config.initialValue };
    }, {} as Record<keyof PredictionPayload, number>);
=======
import { BASELINE_2023, inputConfigs } from "@/config/inputConfig";

const Demo = () => {
  const [params, setParams] = useState(() => {
    return inputConfigs.reduce((acc, config) => {
      return { ...acc, [config.id]: config.initialValue };
    }, {} as Record<keyof Omit<PredictionPayload, 'year'>, number>);
>>>>>>> 84a8efb (refactored version)
  });
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState<number | null>(null);

  const percentChange =
    prediction !== null
      ? (((prediction - BASELINE_2023) / BASELINE_2023) * 100).toFixed(1)
      : null;

  const isIncrease = prediction !== null ? prediction > BASELINE_2023 : false;
  

<<<<<<< HEAD
  const handleChange = (key: keyof PredictionPayload, value: number) => {
    setParams((prev) => ({ ...prev, [key]: value }));
=======
  const handleChange = (id: keyof Omit<PredictionPayload, 'year'>, value: string) => {
    setParams((prev) => ({
      ...prev,
      [id]: value === "" ? 0 : Number(value),
    }));
>>>>>>> 84a8efb (refactored version)
  };

  const handlePredict = async () => {
    setIsLoading(true);
    setPrediction(null);
    try {
      const payload = {
<<<<<<< HEAD
        ...params,
        population: params.population,
        gdp: params.gdp,
=======
        year: 2023,
        population: params.population * 1_000_000,
        gdp: params.gdp * 100_000_000_000,
        ...Object.fromEntries(
          Object.entries(params).filter(([key]) => 
            key !== "population" && key !== "gdp"
          )
        ),
>>>>>>> 84a8efb (refactored version)
      };

      const response = await axios.post<PredictionResponse>(
        "http://localhost:8000/predict",
        payload,
        {
          headers: { "Content-Type": "application/json" },
        }
      );

      if (response.data && typeof response.data.prediction === "number") {
        setPrediction(Number(response.data.prediction));
        toast({
          title: "Prediction Success",
          description: `Predicted CO₂ emissions: ${response.data.prediction.toFixed(1)} Mt`,
        });
      } else {
        throw new Error("Invalid response format");
      }
    } catch (err: any) {
      toast({
        title: "Prediction Failed",
        description: err?.message || "An error occurred",
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
            <CardTitle className="text-slate-800 text-3xl mb-1 tracking-tight">
              Thailand CO₂ Emissions Prediction
            </CardTitle>
            <CardDescription className="mb-2">
              Enter values below and run the model to forecast total annual CO₂ emissions.
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
<<<<<<< HEAD
              <div className="grid gap-6">
                {sliderConfigs.map((config) => (
                  <div key={config.id}>
                    <div className="flex justify-between mb-1">
                      <Label htmlFor={config.id}>{config.label}{config.unit ? ` (${config.unit})` : ''}</Label>
                      <span className="font-mono tabular-nums text-primary">
                        {params[config.id].toFixed(config.step < 1 ? 3 : 0)}
                      </span>
                    </div>
                    <Slider
                      id={config.id}
                      min={config.min}
                      max={config.max}
                      step={config.step}
                      value={[params[config.id]]}
                      onValueChange={(v) => handleChange(config.id, v[0])}
                    />
                    <div className="flex justify-between text-xs mt-0.5 text-muted-foreground">
                      <span>{config.min}</span>
                      <span>{config.max}</span>
                    </div>
=======
              <div className="grid gap-6 md:grid-cols-2">
                {inputConfigs.map((config) => (
                  <div key={config.id} className="space-y-2">
                    <Label htmlFor={config.id}>
                      {config.label}
                      {config.unit && <span className="text-muted-foreground ml-1">({config.unit})</span>}
                    </Label>
                    <Input
                      id={config.id}
                      type="number"
                      step="any"
                      value={params[config.id]}
                      onChange={(e) => handleChange(config.id, e.target.value)}
                      placeholder={config.initialValue.toString()}
                    />
>>>>>>> 84a8efb (refactored version)
                  </div>
                ))}
              </div>

              <Button
                type="submit"
                className="w-full text-lg py-5 bg-blue-600 hover:bg-blue-700 transition-all"
                disabled={isLoading}
              >
                {isLoading ? (
                  <>
                    <svg
                      className="animate-spin mr-3 h-5 w-5 text-white"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                      />
                    </svg>
                    Forecasting...
                  </>
                ) : (
                  <>Predict CO₂ Emissions</>
                )}
              </Button>
            </form>

            {prediction !== null && (
              <motion.div
                initial={{ opacity: 0, scale: 0.97 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.4 }}
                className="mt-8"
              >
                <Card className="w-full bg-gradient-to-br from-blue-100 via-indigo-50 to-green-100 py-6 px-4 border-blue-100 rounded-lg shadow-lg animate-[fade-in_0.8s]">
                  <div className="text-center">
                    <div className="text-gray-500 mb-1 text-sm">
                      Predicted Emissions (Current Configuration)
                    </div>
                    <div className="text-5xl font-bold text-blue-800 mb-2 flex items-center justify-center">
                      {prediction.toFixed(1)}
                      <span className="text-lg font-normal text-blue-600 ml-1">
                        Mt CO₂
                      </span>
                    </div>
                    <div className="flex justify-center items-center gap-2 text-sm mt-1">
                      <span className="text-gray-700">vs 2023 baseline:</span>
                      <span
                        className={`font-medium flex items-center ${
                          isIncrease ? "text-red-500" : "text-green-600"
                        }`}
                      >
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
                      <span className="text-muted-foreground ml-2">
                        ({BASELINE_2023} Mt in 2023)
                      </span>
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