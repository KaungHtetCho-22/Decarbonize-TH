import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts";
import { FeatureImportance } from "@/types/modelTypes";

interface ModelFeatureImportanceProps {
  features: FeatureImportance[];
}

const ModelFeatureImportance = ({ features }: ModelFeatureImportanceProps) => (
  <Card>
    <CardHeader>
      <CardTitle className="text-green-700">Feature Importance</CardTitle>
      <CardDescription>
        Top 10 most influential features in the model
      </CardDescription>
    </CardHeader>
    <CardContent>
      <div className="h-[350px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            layout="vertical"
            data={features}
            margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis dataKey="feature" type="category" width={150} />
            <Tooltip />
            <Bar dataKey="importance" fill="#22c55e" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </CardContent>
  </Card>
);

export default ModelFeatureImportance;
