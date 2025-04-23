
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
import { BASELINE_2023, inputConfigs } from "@/config/inputConfig";
import {
Select,
SelectTrigger,
SelectValue,
SelectContent,
SelectItem,
} from "@/components/ui/select";

const MODEL_OPTIONS = [
{ value: "xgboost", label: "XGBoost" },
{ value: "random_forest", label: "Random Forest" },
{ value: "lightgbm", label: "LightGBM" },
{ value: "catboost", label: "CatBoost" },
{ value: "gradient_boosting", label: "Gradient Boosting" },
];

const getModelLabel = (value: string) =>
MODEL_OPTIONS.find((m) => m.value === value)?.label || value;

const Demo = () => {
const [params, setParams] = useState(() => {
  return inputConfigs.reduce((acc, config) => {
    return { ...acc, [config.id]: config.initialValue };
  }, {} as Record<keyof Omit<PredictionPayload, 'year'>, number>);
});
const [isLoading, setIsLoading] = useState(false);
const [prediction, setPrediction] = useState<number | null>(null);
const [selectedModel, setSelectedModel] = useState("xgboost");

const percentChange =
  prediction !== null
    ? (((prediction - BASELINE_2023) / BASELINE_2023) * 100).toFixed(1)
    : null;

const isIncrease = prediction !== null ? prediction > BASELINE_2023 : false;

const handleChange = (id: keyof Omit<PredictionPayload, 'year'>, value: string) => {
  setParams((prev) => ({
    ...prev,
    [id]: value === "" ? 0 : Number(value),
  }));
};

const handlePredict = async () => {
  setIsLoading(true);
  setPrediction(null);
  try {
    const payload = {
      year: 2023,
      population: params.population * 1_000_000,
      gdp: params.gdp *  100_000_000_000,
      ...Object.fromEntries(
        Object.entries(params).filter(([key]) =>
          key !== "population" && key !== "gdp"
        )
      ),
    };

    // Updated: Add model_name as query parameter
    const response = await axios.post<PredictionResponse>(
      `http://54.91.195.11:8000/predict?model_name=${selectedModel}`,
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
  <div className="min-h-screen bg-background pt-16 px-2 pb-24 animate-fade-in">
    <motion.div
      initial={{ opacity: 0, translateY: 24 }}
      animate={{ opacity: 1, translateY: 0 }}
      transition={{ duration: 0.7 }}
      className="max-w-3xl mx-auto"
    >
      <Card className="bg-card card-green shadow-md border border-border">
        <CardHeader>
          <CardTitle className="text-foreground text-3xl mb-1 tracking-tight">
            Thailand CO₂ Emissions Prediction
          </CardTitle>
          <CardDescription className="mb-2">
            Enter values below and run the model to forecast total annual CO₂ emissions.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {/* MODEL SELECTION DROPDOWN */}
          <form
            className="space-y-6"
            onSubmit={(e) => {
              e.preventDefault();
              handlePredict();
            }}
          >
            <div className="mb-2 space-y-2">
              <Label htmlFor="model-select" className="text-foreground">
                Model Selection
              </Label>
              <Select
                value={selectedModel}
                onValueChange={(val) => setSelectedModel(val)}
                name="model"
              >
                <SelectTrigger id="model-select" className="w-full">
                  <SelectValue placeholder="Select model..." />
                </SelectTrigger>
                <SelectContent className="bg-popover z-50">
                  {MODEL_OPTIONS.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value}>
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
              {inputConfigs.map((config) => (
                <div key={config.id} className="space-y-2">
                  <Label htmlFor={config.id} className="text-foreground">
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
                </div>
              ))}
            </div>

            <Button
              type="submit"
              className="w-full text-lg py-5 btn-green bg-primary hover:bg-primary/90 transition-all"
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
              <Card className="w-full card-green bg-secondary py-6 px-4 border border-border rounded-xl shadow-lg animate-[fade-in_0.8s]">
                <div className="text-center">
                  <div className="text-muted-foreground mb-1 text-sm">
                    Predicted Emissions (Current Configuration)
                  </div>
                  <div className="text-5xl font-bold text-primary mb-2 flex items-center justify-center">
                    {prediction.toFixed(1)}
                    <span className="text-lg font-normal text-primary ml-1">
                      Mt CO₂
                    </span>
                  </div>
                  <div className="flex justify-center items-center gap-2 text-sm mt-1">
                    <span className="text-foreground">vs 2023 baseline:</span>
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
                  <div className="mt-4">
                    <span className="text-foreground font-medium">
                      Model Used:{" "}
                      <span className="font-semibold">{getModelLabel(selectedModel)}</span>
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