
import React, { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { featureData, formatFeatureName } from "@/utils/featureData";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ArrowUpIcon, ArrowDownIcon } from "lucide-react";

// Sort helpers
type SortBy = "correlation" | "name";
type SortDir = "asc" | "desc";

interface FeatureCorrelationTableProps {
  className?: string;
}

const getCorrelationBadgeColor = (correlation: number): string => {
  if (correlation >= 0.7) return "bg-green-700 text-white";
  if (correlation >= 0.4) return "bg-green-500 text-white";
  if (correlation >= 0.2) return "bg-green-200 text-green-900";
  if (correlation > -0.2) return "bg-gray-100 text-gray-900";
  if (correlation > -0.5) return "bg-red-100 text-red-900";
  return "bg-red-400 text-white";
};

const truncateText = (str: string, max: number = 22) => (str.length > max ? str.slice(0, max - 1) + "…" : str);

const FeatureCorrelationTable: React.FC<FeatureCorrelationTableProps> = ({ className }) => {
  const [sortBy, setSortBy] = useState<SortBy>("correlation");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const sortedFeatures = [...featureData].sort((a, b) => {
    if (sortBy === "name") {
      const aName = a.name.toLowerCase();
      const bName = b.name.toLowerCase();
      return sortDir === "asc" ? aName.localeCompare(bName) : bName.localeCompare(aName);
    } else {
      // correlation
      return sortDir === "asc" ? a.correlation - b.correlation : b.correlation - a.correlation;
    }
  });

  const handleSort = (column: SortBy) => {
    if (column === sortBy) {
      setSortDir(prev => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortBy(column);
      setSortDir("desc");
    }
  };

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="text-green-700">Feature Correlation with CO₂</CardTitle>
        <CardDescription>Sortable, interactive table of each feature's correlation with CO₂ emissions.</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="w-full overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="cursor-pointer" onClick={() => handleSort("name")}>
                  Feature
                  <span className="inline-block align-middle ml-1">
                    {sortBy === "name" && (sortDir === "asc" ? <ArrowUpIcon className="inline h-4 w-4" /> : <ArrowDownIcon className="inline h-4 w-4" />)}
                  </span>
                </TableHead>
                <TableHead className="text-center cursor-pointer" onClick={() => handleSort("correlation")}>
                  Correlation
                  <span className="inline-block align-middle ml-1">
                    {sortBy === "correlation" && (sortDir === "asc" ? <ArrowUpIcon className="inline h-4 w-4" /> : <ArrowDownIcon className="inline h-4 w-4" />)}
                  </span>
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {sortedFeatures.map((feature, idx) => (
                <TableRow key={feature.name} className="hover:bg-green-50 group">
                  <TableCell className="font-medium p-3 align-middle whitespace-nowrap max-w-xs">
                    <span title={formatFeatureName(feature.name)} className="truncate block max-w-[180px]">
                      {truncateText(formatFeatureName(feature.name), 32)}
                    </span>
                  </TableCell>
                  <TableCell className="text-center p-3 align-middle">
                    <Badge className={getCorrelationBadgeColor(feature.correlation)}>
                      {feature.correlation.toFixed(2)}
                    </Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
};
export default FeatureCorrelationTable;

