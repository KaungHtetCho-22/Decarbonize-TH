
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
} from 'recharts';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ChartContainer, ChartTooltip } from "@/components/ui/chart";

// Updated split numbers as requested
const datasetSplitData = [
  { split: "Global Train (World excl. ASEAN + Thailand)", rows: 15392 },
  { split: "ASEAN Train (60%)", rows: 399 },
  { split: "ASEAN Val (40%)", rows: 267 },
  { split: "Thailand Test", rows: 74 }
];

const COLORS = ['#16a34a', '#22c55e', '#4ade80', '#86efac'];

const DatasetSplitSummary = () => {
  const total = datasetSplitData.reduce((sum, item) => sum + item.rows, 0);
  
  const formattedData = datasetSplitData.map(item => ({
    ...item,
    percentage: (item.rows / total * 100).toFixed(1) + '%'
  }));

  return (
    <Card className="mb-8">
      <CardHeader>
        <CardTitle className="text-2xl text-green-700">Train-Validation-Test Split</CardTitle>
        <CardDescription>
          Summary of data allocation for model training, validation, and testing
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="h-[250px]">
            <ChartContainer
              config={{
                rows: {
                  label: "Number of Rows",
                  color: "#16a34a"
                }
              }}
            >
              <PieChart>
                <Pie
                  data={datasetSplitData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={95}
                  fill="#8884d8"
                  paddingAngle={2}
                  dataKey="rows"
                  nameKey="split"
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                  labelLine={false}
                >
                  {datasetSplitData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <ChartTooltip 
                  formatter={(value: number) => `${value} rows`}
                />
              </PieChart>
            </ChartContainer>
          </div>
          
          <div className="flex items-center">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Data Split</TableHead>
                  <TableHead className="text-right">Rows</TableHead>
                  <TableHead className="text-right">Percentage</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {formattedData.map((item, index) => (
                  <TableRow key={index}>
                    <TableCell className="font-medium">{item.split}</TableCell>
                    <TableCell className="text-right">{item.rows}</TableCell>
                    <TableCell className="text-right">{item.percentage}</TableCell>
                  </TableRow>
                ))}
                <TableRow className="font-bold">
                  <TableCell>Total</TableCell>
                  <TableCell className="text-right">{total}</TableCell>
                  <TableCell className="text-right">100%</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default DatasetSplitSummary;
