export interface ModelPrediction {
    year: number;
    actual: number;
    predicted: number;
  }
  
  export interface FeatureImportance {
    feature: string;
    importance: number;
  }
  
  export interface ModelMetrics {
    name: string;
    id: string;
    predictions: ModelPrediction[];
    featureImportance: FeatureImportance[];
    metrics: {
      rmse: number;
      mae: number;
      r2: number;
    };
  }
  
  export interface ModelSummary {
    name: string;
    id: string;
    metrics: {
      rmse: number;
      mae: number;
      r2: number;
    };
  }