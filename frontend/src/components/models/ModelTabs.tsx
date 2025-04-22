import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ModelMetrics } from "@/types/modelTypes";
import ModelPredictionChart from "./ModelPredictionChart";
import ModelFeatureImportance from "./ModelFeatureImportance";
import ModelMetricsTable from "./ModelMetricsTable";
import { Card } from "@/components/ui/card";

interface ModelTabsProps {
  models: ModelMetrics[];
  bestModelId: string;
}

const ModelTabs = ({ models, bestModelId }: ModelTabsProps) => {
  const [selectedModel, setSelectedModel] = useState<string>(bestModelId);

  return (
    <Tabs value={selectedModel} onValueChange={setSelectedModel}>
      <div className="flex justify-center mb-6 overflow-x-auto">
        <TabsList className="h-auto flex-wrap">
          {models.map((model) => (
            <TabsTrigger 
              key={model.id} 
              value={model.id}
              className="px-4 py-2 data-[state=active]:bg-green-600 data-[state=active]:text-white"
            >
              {model.name}
            </TabsTrigger>
          ))}
        </TabsList>
      </div>
      
      {models.map((model) => (
        <TabsContent key={model.id} value={model.id}>
          <div className="grid md:grid-cols-2 gap-8">
            <ModelPredictionChart predictions={model.predictions} />
            <ModelFeatureImportance features={model.featureImportance} />
            <ModelMetricsTable 
              metrics={model.metrics} 
              isBest={model.id === bestModelId} 
              className="md:col-span-2" 
            />
          </div>
        </TabsContent>
      ))}
    </Tabs>
  );
};

export default ModelTabs;