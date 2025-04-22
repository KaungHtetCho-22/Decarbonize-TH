
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LabelList
} from 'recharts';
import { missingValuesByFeature } from "@/utils/thailandData";
import { ChartContainer, ChartTooltip } from "@/components/ui/chart";

const MissingValuesDiagnostic = () => {
  return (
    <Card className="mb-8">
      <CardHeader>
        <CardTitle className="text-2xl text-green-700">Missing Value Ratio (Selected Columns) Before vs After Fill+Backfill</CardTitle>
        <CardDescription>
          Effectiveness of forward-fill and backward-fill in reducing missing data across key features
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-[400px]">
          <ChartContainer
            config={{
              before: {
                label: "Before Fill",
                color: "#f97316"
              },
              after: {
                label: "After Fill",
                color: "#16a34a"
              }
            }}
          >
            <BarChart
              data={missingValuesByFeature}
              margin={{ top: 20, right: 30, left: 20, bottom: 120 }}
              barGap={4}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="feature" 
                angle={-45} 
                textAnchor="end" 
                height={120}
                interval={0}
                tick={{ fontSize: 12 }}
              />
              <YAxis
                tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                domain={[0, 1]}
              />
              <ChartTooltip
                formatter={(value) => `${(Number(value) * 100).toFixed(2)}%`}
              />
              <Legend />
              <Bar dataKey="before" fill="#f97316" name="Before Fill">
                <LabelList dataKey="before" position="top" formatter={(value) => (value > 0.1 ? `${(value * 100).toFixed(0)}%` : '')} />
              </Bar>
              <Bar dataKey="after" fill="#16a34a" name="After Fill">
                <LabelList dataKey="after" position="top" formatter={(value) => (value > 0.05 ? `${(value * 100).toFixed(0)}%` : '')} />
              </Bar>
            </BarChart>
          </ChartContainer>
        </div>
      </CardContent>
    </Card>
  );
};

export default MissingValuesDiagnostic;
