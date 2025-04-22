
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface ModelMetricsTableProps {
  metrics: {
    rmse: number;
    mae: number;
    r2: number;
  };
  isBest: boolean;
  className?: string;
}

const ModelMetricsTable = ({ metrics, isBest, className }: ModelMetricsTableProps) => (
  <Card className={cn(className)}>
    <CardHeader className="flex flex-row items-center justify-between">
      <CardTitle className="text-green-700">Performance Metrics</CardTitle>
      {isBest && (
        <Badge variant="secondary" className="bg-green-100 text-green-800">
          Best Performer
        </Badge>
      )}
    </CardHeader>
    <CardContent>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Metric</TableHead>
            <TableHead>Value</TableHead>
            <TableHead>Description</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          <TableRow>
            <TableCell className="font-medium">RMSE</TableCell>
            <TableCell>{metrics.rmse.toFixed(4)}</TableCell>
            <TableCell>Root Mean Square Error - Lower is better</TableCell>
          </TableRow>
          <TableRow>
            <TableCell className="font-medium">MAE</TableCell>
            <TableCell>{metrics.mae.toFixed(4)}</TableCell>
            <TableCell>Mean Absolute Error - Lower is better</TableCell>
          </TableRow>
          <TableRow>
            <TableCell className="font-medium">R²</TableCell>
            <TableCell>{metrics.r2.toFixed(4)}</TableCell>
            <TableCell>Coefficient of Determination - Closer to 1 is better</TableCell>
          </TableRow>
        </TableBody>
      </Table>
    </CardContent>
  </Card>
);

export default ModelMetricsTable;