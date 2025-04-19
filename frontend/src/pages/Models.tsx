
import { useState } from "react";
import { motion } from "framer-motion";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { 
  Tabs, 
  TabsContent, 
  TabsList, 
  TabsTrigger 
} from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  LineChart,
  Line,
  Cell
} from "recharts";
import { Button } from "@/components/ui/button";
import { BarChart3, Sparkles, LineChart as LineChartIcon, Network, TrendingUp, ArrowRight, Info } from "lucide-react";

const Models = () => {
  const [selectedModel, setSelectedModel] = useState("xgboost");
  
  // Model comparison data
  const modelMetrics = [
    { name: "XGBoost", mae: 1.82, rmse: 2.41, r2: 0.927, trainTime: 63, id: "xgboost" },
    { name: "Random Forest", mae: 2.14, rmse: 2.76, r2: 0.903, trainTime: 49, id: "random-forest" },
    { name: "Neural Network", mae: 2.37, rmse: 3.02, r2: 0.886, trainTime: 128, id: "neural-network" },
    { name: "Linear Regression", mae: 3.69, rmse: 4.25, r2: 0.822, trainTime: 5, id: "linear-regression" },
    { name: "SVR", mae: 2.91, rmse: 3.62, r2: 0.865, trainTime: 87, id: "svr" }
  ];

  // Performance metrics explained
  const metricsExplained = [
    {
      name: "MAE (Mean Absolute Error)",
      description: "The average of the absolute differences between predictions and actual values. Lower is better.",
      relevance: "Provides an easy-to-understand measure of error magnitude in the original unit (Mt CO₂)."
    },
    {
      name: "RMSE (Root Mean Square Error)",
      description: "Square root of the average of squared differences between predictions and actual values. Lower is better.",
      relevance: "Penalizes large errors more heavily than small ones, making it sensitive to outliers."
    },
    {
      name: "R² (Coefficient of Determination)",
      description: "Represents the proportion of variance in the dependent variable explained by the model. Closer to 1 is better.",
      relevance: "Indicates how well the model explains the variation in carbon emissions data."
    },
    {
      name: "Training Time",
      description: "Time taken to train the model in seconds.",
      relevance: "Reflects computational efficiency, important for retraining and deployment."
    }
  ];
  
  // Feature importance data
  const featureImportance = {
    "xgboost": [
      { feature: "Energy Consumption", importance: 27.8 },
      { feature: "Transport Activity", importance: 18.5 },
      { feature: "Industrial Output", importance: 16.2 },
      { feature: "Energy Intensity", importance: 13.1 },
      { feature: "GDP", importance: 9.4 },
      { feature: "Population", importance: 7.9 },
      { feature: "Climate Factor", importance: 5.3 },
      { feature: "Agricultural Production", importance: 1.8 }
    ],
    "random-forest": [
      { feature: "Energy Consumption", importance: 25.6 },
      { feature: "Transport Activity", importance: 19.2 },
      { feature: "Industrial Output", importance: 14.8 },
      { feature: "Energy Intensity", importance: 12.5 },
      { feature: "GDP", importance: 10.3 },
      { feature: "Population", importance: 8.7 },
      { feature: "Climate Factor", importance: 6.2 },
      { feature: "Agricultural Production", importance: 2.7 }
    ],
    "neural-network": [
      { feature: "Energy Consumption", importance: 29.3 },
      { feature: "Transport Activity", importance: 17.6 },
      { feature: "Industrial Output", importance: 15.4 },
      { feature: "Energy Intensity", importance: 11.8 },
      { feature: "GDP", importance: 8.9 },
      { feature: "Population", importance: 7.2 },
      { feature: "Climate Factor", importance: 6.8 },
      { feature: "Agricultural Production", importance: 3.0 }
    ],
    "linear-regression": [
      { feature: "Energy Consumption", importance: 31.5 },
      { feature: "Transport Activity", importance: 16.8 },
      { feature: "Industrial Output", importance: 14.2 },
      { feature: "Energy Intensity", importance: 10.3 },
      { feature: "GDP", importance: 11.7 },
      { feature: "Population", importance: 8.5 },
      { feature: "Climate Factor", importance: 4.2 },
      { feature: "Agricultural Production", importance: 2.8 }
    ],
    "svr": [
      { feature: "Energy Consumption", importance: 26.9 },
      { feature: "Transport Activity", importance: 18.2 },
      { feature: "Industrial Output", importance: 15.7 },
      { feature: "Energy Intensity", importance: 12.8 },
      { feature: "GDP", importance: 9.1 },
      { feature: "Population", importance: 7.5 },
      { feature: "Climate Factor", importance: 6.3 },
      { feature: "Agricultural Production", importance: 3.5 }
    ]
  };
  
  // Prediction accuracy for each model
  const predictionAccuracy = {
    "xgboost": [
      { year: "2017", actual: 274.6, predicted: 273.2 },
      { year: "2018", actual: 283.4, predicted: 280.9 },
      { year: "2019", actual: 291.7, predicted: 293.8 },
      { year: "2020", actual: 267.8, predicted: 271.1 },
      { year: "2021", actual: 281.9, predicted: 283.5 }
    ],
    "random-forest": [
      { year: "2017", actual: 274.6, predicted: 271.9 },
      { year: "2018", actual: 283.4, predicted: 279.5 },
      { year: "2019", actual: 291.7, predicted: 295.2 },
      { year: "2020", actual: 267.8, predicted: 274.6 },
      { year: "2021", actual: 281.9, predicted: 285.1 }
    ],
    "neural-network": [
      { year: "2017", actual: 274.6, predicted: 270.2 },
      { year: "2018", actual: 283.4, predicted: 277.6 },
      { year: "2019", actual: 291.7, predicted: 294.3 },
      { year: "2020", actual: 267.8, predicted: 275.8 },
      { year: "2021", actual: 281.9, predicted: 278.5 }
    ],
    "linear-regression": [
      { year: "2017", actual: 274.6, predicted: 268.1 },
      { year: "2018", actual: 283.4, predicted: 276.3 },
      { year: "2019", actual: 291.7, predicted: 298.4 },
      { year: "2020", actual: 267.8, predicted: 280.2 },
      { year: "2021", actual: 281.9, predicted: 273.6 }
    ],
    "svr": [
      { year: "2017", actual: 274.6, predicted: 270.5 },
      { year: "2018", actual: 283.4, predicted: 279.1 },
      { year: "2019", actual: 291.7, predicted: 295.6 },
      { year: "2020", actual: 267.8, predicted: 274.1 },
      { year: "2021", actual: 281.9, predicted: 284.4 }
    ]
  };
  
  // Radar chart data for model comparison across dimensions
  const radarData = [
    { dimension: "Accuracy", "XGBoost": 9.2, "Random Forest": 8.7, "Neural Network": 8.4, "Linear Regression": 6.8, "SVR": 8.1 },
    { dimension: "Speed", "XGBoost": 8.2, "Random Forest": 8.5, "Neural Network": 6.3, "Linear Regression": 9.5, "SVR": 7.4 },
    { dimension: "Interpretability", "XGBoost": 7.8, "Random Forest": 8.3, "Neural Network": 5.4, "Linear Regression": 9.7, "SVR": 6.8 },
    { dimension: "Generalization", "XGBoost": 8.9, "Random Forest": 8.4, "Neural Network": 8.6, "Linear Regression": 7.3, "SVR": 8.2 },
    { dimension: "Robustness", "XGBoost": 9.0, "Random Forest": 8.8, "Neural Network": 8.3, "Linear Regression": 6.9, "SVR": 8.5 },
  ];
  
  // Hyperparameter tuning results
  const hyperparameterData = [
    { 
      model: "XGBoost", 
      parameters: [
        { name: "max_depth", value: 6 },
        { name: "learning_rate", value: 0.1 },
        { name: "n_estimators", value: 200 },
        { name: "subsample", value: 0.8 },
        { name: "colsample_bytree", value: 0.8 }
      ] 
    },
    { 
      model: "Random Forest", 
      parameters: [
        { name: "n_estimators", value: 250 },
        { name: "max_depth", value: 12 },
        { name: "min_samples_split", value: 5 },
        { name: "min_samples_leaf", value: 2 },
        { name: "max_features", value: "sqrt" }
      ] 
    },
    { 
      model: "Neural Network", 
      parameters: [
        { name: "hidden_layers", value: "64, 32, 16" },
        { name: "activation", value: "relu" },
        { name: "optimizer", value: "adam" },
        { name: "learning_rate", value: 0.001 },
        { name: "batch_size", value: 32 }
      ] 
    },
    { 
      model: "SVR", 
      parameters: [
        { name: "kernel", value: "rbf" },
        { name: "C", value: 10 },
        { name: "gamma", value: 0.1 },
        { name: "epsilon", value: 0.1 },
      ] 
    }
  ];

  // XGBoost model architecture (simplified)
  const xgboostArchitecture = [
    { stage: "Input Layer", nodes: "Feature Vector (48 features)", description: "Standardized input features" },
    { stage: "Tree 1", nodes: "Depth: 6", description: "First decision tree in the ensemble" },
    { stage: "Tree 2", nodes: "Depth: 6", description: "Second tree, focuses on residuals from Tree 1" },
    { stage: "Tree n", nodes: "Total Trees: 200", description: "Each subsequent tree learns from previous trees' errors" },
    { stage: "Output", nodes: "Weighted Sum", description: "Final prediction combines all trees' outputs" }
  ];
  
  const COLORS = ['#22c55e', '#16a34a', '#4ade80', '#86efac', '#bbf7d0'];

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
            <BarChart3 className="h-8 w-8 text-green-600" />
          </div>
          <h1 className="text-4xl font-bold text-green-800 mb-6">Predictive Models</h1>
          <p className="text-lg text-gray-700 max-w-3xl mx-auto mb-8">
            Explore the machine learning models we've developed to predict carbon emissions in Thailand, 
            and learn how we selected the best performing approach.
          </p>
        </div>
        
        <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
          {[...Array(20)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute bg-green-600 opacity-10"
              style={{
                height: 2 + Math.random() * 40,
                width: 2,
                top: Math.random() * 100 + '%',
                left: Math.random() * 100 + '%',
              }}
              animate={{
                height: [2 + Math.random() * 40, 60 + Math.random() * 80, 2 + Math.random() * 40],
              }}
              transition={{
                duration: 2 + Math.random() * 3,
                repeat: Infinity,
                delay: Math.random() * 5,
              }}
            />
          ))}
        </div>
      </motion.section>
      
      <section className="py-12">
        <div className="section-container">
          <h2 className="section-title">Model Comparison</h2>
          <p className="text-gray-700 mb-8">
            We evaluated several machine learning algorithms to find the best approach for predicting carbon emissions.
            The metrics below show how each model performed on our test dataset.
          </p>
          
          <div className="grid md:grid-cols-2 gap-8 mb-12">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
              viewport={{ once: true }}
            >
              <Card className="overflow-hidden">
                <CardHeader>
                  <CardTitle className="text-green-700">Performance Metrics</CardTitle>
                  <CardDescription>
                    Comparing error metrics and training times across models
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[350px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={modelMetrics}
                        margin={{ top: 20, right: 30, left: 20, bottom: 70 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis 
                          dataKey="name" 
                          angle={0}
                          textAnchor="middle"
                        />
                        <YAxis 
                          yAxisId="left"
                          orientation="left"
                          label={{ value: 'Error (MT CO₂)', angle: -90, position: 'insideLeft' }}
                        />
                        <YAxis 
                          yAxisId="right" 
                          orientation="right"
                          label={{ value: 'R² Score', angle: 90, position: 'insideRight' }}
                          domain={[0, 1]}
                        />
                        <Tooltip />
                        <Legend verticalAlign="top" wrapperStyle={{ paddingBottom: 10 }} />
                        <Bar yAxisId="left" dataKey="rmse" name="RMSE" fill="#ef4444" />
                        <Bar yAxisId="left" dataKey="mae" name="MAE" fill="#f97316" />
                        <Bar yAxisId="right" dataKey="r2" name="R² Score" fill="#22c55e" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="flex items-center justify-center mt-4 text-sm text-gray-500">
                    <Info size={16} className="mr-2 text-green-600" />
                    Lower MAE and RMSE values indicate better performance, while higher R² is better
                  </div>
                </CardContent>
              </Card>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
              viewport={{ once: true }}
            >
              <Card className="overflow-hidden">
                <CardHeader>
                  <CardTitle className="text-green-700">Model Characteristics</CardTitle>
                  <CardDescription>
                    Multi-dimensional comparison of model capabilities
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[350px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <RadarChart outerRadius={90} data={radarData}>
                        <PolarGrid />
                        <PolarAngleAxis dataKey="dimension" />
                        <PolarRadiusAxis angle={30} domain={[0, 10]} />
                        <Radar name="XGBoost" dataKey="XGBoost" stroke="#22c55e" fill="#22c55e" fillOpacity={0.6} />
                        <Radar name="Random Forest" dataKey="Random Forest" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.6} />
                        <Radar name="Neural Network" dataKey="Neural Network" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.6} />
                        <Radar name="Linear Regression" dataKey="Linear Regression" stroke="#f97316" fill="#f97316" fillOpacity={0.6} />
                        <Radar name="SVR" dataKey="SVR" stroke="#ec4899" fill="#ec4899" fillOpacity={0.6} />
                        <Tooltip />
                        <Legend />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="flex items-center justify-center mt-4 text-sm text-gray-500">
                    <Info size={16} className="mr-2 text-green-600" />
                    Scores from 0-10, with 10 being the best performance in that dimension
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </div>
          
          <div className="bg-green-50 p-5 rounded-md mb-12">
            <div className="flex items-start mb-3">
              <Sparkles className="h-6 w-6 text-green-600 mr-3 mt-0.5 flex-shrink-0" />
              <div>
                <h3 className="font-semibold text-green-800 text-lg">Best Performing Model: XGBoost</h3>
                <p className="text-gray-700">
                  After extensive testing and cross-validation, XGBoost was selected as our primary model due to its 
                  superior accuracy (lowest MAE and RMSE, highest R²), good interpretability, and excellent 
                  balance between performance and computational efficiency.
                </p>
              </div>
            </div>
          </div>
          
          <div className="mb-8">
            <Card>
              <CardHeader>
                <CardTitle className="text-green-700">Performance Metrics Explained</CardTitle>
                <CardDescription>
                  Understanding the evaluation criteria used to select the best model
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-[250px]">Metric</TableHead>
                      <TableHead>Description</TableHead>
                      <TableHead className="hidden md:table-cell">Relevance</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {metricsExplained.map((metric, index) => (
                      <TableRow key={index}>
                        <TableCell className="font-medium">{metric.name}</TableCell>
                        <TableCell>{metric.description}</TableCell>
                        <TableCell className="hidden md:table-cell">{metric.relevance}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>
      
      <section className="py-12 bg-green-50">
        <div className="section-container">
          <h2 className="section-title">Model Details</h2>
          <p className="text-gray-700 mb-8">
            Explore the features, predictions, and configurations of our machine learning models.
          </p>
          
          <Tabs value={selectedModel} onValueChange={setSelectedModel}>
            <div className="flex justify-center mb-6 overflow-x-auto">
              <TabsList className="h-auto flex-wrap">
                {modelMetrics.map(model => (
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
            
            {modelMetrics.map(model => (
              <TabsContent key={model.id} value={model.id}>
                <div className="grid md:grid-cols-2 gap-8">
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-green-700">Feature Importance</CardTitle>
                      <CardDescription>
                        Relative importance of each feature in the {model.name} model
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="h-[350px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={featureImportance[model.id]}
                            layout="vertical"
                            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis type="number" domain={[0, 30]} unit="%" />
                            <YAxis 
                              dataKey="feature" 
                              type="category" 
                              width={150}
                            />
                            <Tooltip formatter={(value) => [`${value}%`, 'Importance']} />
                            <Bar dataKey="importance" fill="#22c55e">
                              {featureImportance[model.id].map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-green-700">Prediction Accuracy</CardTitle>
                      <CardDescription>
                        Comparison of actual vs. predicted values for test years
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="h-[350px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart
                            data={predictionAccuracy[model.id]}
                            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="year" />
                            <YAxis domain={[260, 300]} label={{ value: 'CO₂ Emissions (MT)', angle: -90, position: 'insideLeft' }} />
                            <Tooltip />
                            <Legend />
                            <Line 
                              type="monotone" 
                              dataKey="actual" 
                              name="Actual" 
                              stroke="#3b82f6" 
                              strokeWidth={2} 
                              dot={{ r: 6 }}
                              activeDot={{ r: 8 }}
                            />
                            <Line 
                              type="monotone" 
                              dataKey="predicted" 
                              name="Predicted" 
                              stroke="#22c55e" 
                              strokeWidth={2} 
                              dot={{ r: 6 }}
                              activeDot={{ r: 8 }}
                              strokeDasharray="5 5"
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </CardContent>
                  </Card>
                </div>
                
                <div className="mt-8">
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-green-700">Model Configuration</CardTitle>
                      <CardDescription>
                        Optimal hyperparameters used in the {model.name} model
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      {model.id !== "linear-regression" && (
                        <div>
                          <Table>
                            <TableHeader>
                              <TableRow>
                                <TableHead>Parameter</TableHead>
                                <TableHead>Value</TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {hyperparameterData.find(data => data.model === model.name)?.parameters.map((param, index) => (
                                <TableRow key={index}>
                                  <TableCell className="font-mono">{param.name}</TableCell>
                                  <TableCell className="font-mono">{param.value}</TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                          
                          {model.id === "xgboost" && (
                            <div className="mt-6">
                              <h4 className="text-sm font-medium text-green-700 mb-3">Model Architecture</h4>
                              <div className="space-y-3">
                                {xgboostArchitecture.map((item, index) => (
                                  <div key={index} className="flex">
                                    <div className="mr-4 flex flex-col items-center">
                                      <div className="w-8 h-8 rounded-full bg-green-600 flex items-center justify-center text-white font-medium">
                                        {index + 1}
                                      </div>
                                      {index < xgboostArchitecture.length - 1 && (
                                        <div className="w-0.5 h-full bg-green-200 my-1"></div>
                                      )}
                                    </div>
                                    <div>
                                      <h5 className="font-medium text-gray-800">{item.stage}</h5>
                                      <p className="text-sm text-gray-600">{item.nodes}</p>
                                      <p className="text-xs text-gray-500">{item.description}</p>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                      
                      {model.id === "linear-regression" && (
                        <div className="text-center py-6">
                          <p className="text-gray-700">
                            Linear Regression uses a simple linear equation to model the relationship between features and emissions.
                            No complex hyperparameters are required for this baseline model.
                          </p>
                          <div className="mt-4 bg-green-50 p-4 rounded-md inline-block text-left">
                            <code className="text-sm text-green-800 font-mono">
                              y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
                            </code>
                            <p className="text-xs text-gray-600 mt-2">
                              Where y is the predicted carbon emission, β₀ is the intercept, 
                              and β₁...βₙ are the coefficients for each feature x₁...xₙ
                            </p>
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>
            ))}
          </Tabs>
        </div>
      </section>
      
      <section className="py-12">
        <div className="section-container">
          <h2 className="section-title">Model Selection Process</h2>
          <p className="text-gray-700 mb-8">
            How we determined that XGBoost was the best model for our carbon emission predictions
          </p>
          
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                title: "Model Training",
                icon: <LineChartIcon className="h-10 w-10 text-green-600 mb-4" />,
                steps: [
                  "Split data into training (70%), validation (15%), and test (15%) sets",
                  "Train multiple models with default hyperparameters",
                  "Evaluate performance on validation set",
                  "Use 5-fold cross-validation to ensure robustness"
                ]
              },
              {
                title: "Model Tuning",
                icon: <Network className="h-10 w-10 text-green-600 mb-4" />,
                steps: [
                  "Perform extensive hyperparameter optimization using grid search",
                  "Use Bayesian optimization for complex models like Neural Networks",
                  "Evaluate each configuration on validation set",
                  "Select best configuration for each model type"
                ]
              },
              {
                title: "Model Evaluation",
                icon: <TrendingUp className="h-10 w-10 text-green-600 mb-4" />,
                steps: [
                  "Evaluate final models on unseen test data",
                  "Compare performance across multiple metrics",
                  "Analyze feature importance and interpretability",
                  "Consider computational requirements and inference speed"
                ]
              }
            ].map((section, index) => (
              <Card key={index} className="h-full flex flex-col">
                <CardHeader>
                  <CardTitle className="text-green-700 flex flex-col items-center">
                    {section.icon}
                    {section.title}
                  </CardTitle>
                </CardHeader>
                <CardContent className="flex-grow">
                  <ol className="space-y-3 list-decimal pl-5">
                    {section.steps.map((step, stepIndex) => (
                      <li key={stepIndex} className="text-gray-700">{step}</li>
                    ))}
                  </ol>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>
      
      <section className="py-12 bg-gradient-to-br from-green-700 to-green-900 text-white">
        <div className="section-container">
          <div className="text-center">
            <h2 className="text-3xl font-bold mb-4">See the Model in Action!</h2>
            <p className="max-w-2xl mx-auto mb-8 text-green-100">
              Try our interactive demo to see how our XGBoost model predicts carbon emissions based on different inputs.
            </p>
            <motion.div
              className="inline-block"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <a 
                href="/demo" 
                className="inline-flex items-center bg-white text-green-700 px-6 py-3 rounded-md font-medium hover:bg-green-50 transition-colors"
              >
                Try the Demo
                <ArrowRight className="ml-2 h-5 w-5" />
              </a>
            </motion.div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Models;
