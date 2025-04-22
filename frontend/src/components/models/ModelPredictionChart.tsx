import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts";
import { ModelPrediction } from "@/types/modelTypes";

interface ModelPredictionChartProps {
  predictions: ModelPrediction[];
}

const ModelPredictionChart = ({ predictions }: ModelPredictionChartProps) => (
  <Card>
    <CardHeader>
      <CardTitle className="text-green-700">Prediction Accuracy</CardTitle>
      <CardDescription>
        Comparison of actual vs. predicted values (2000-2023)
      </CardDescription>
    </CardHeader>
    <CardContent>
      <div className="h-[350px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={predictions} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year" />
            <YAxis 
              domain={['auto', 'auto']}
              label={{ value: 'CO₂ Emissions (MT)', angle: -90, position: 'insideLeft' }} 
            />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="actual"
              name="Actual"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={{ r: 4 }}
              activeDot={{ r: 6 }}
            />
            <Line
              type="monotone"
              dataKey="predicted"
              name="Predicted"
              stroke="#22c55e"
              strokeWidth={2}
              dot={{ r: 4 }}
              activeDot={{ r: 6 }}
              strokeDasharray="5 5"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </CardContent>
  </Card>
);

export default ModelPredictionChart;