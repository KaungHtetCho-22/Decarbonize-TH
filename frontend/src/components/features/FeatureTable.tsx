
import React from 'react';
import { motion } from 'framer-motion';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { featureData, formatFeatureName } from '@/utils/featureData';
import { Badge } from "@/components/ui/badge";

interface FeatureTableProps {
  className?: string;
}

const FeatureTable: React.FC<FeatureTableProps> = ({ className }) => {
  // Sort features by correlation with CO2 (absolute value)
  const sortedFeatures = [...featureData].sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));
  
  // Helper to get the correlation badge color (green=high, gray=low/neg, red=pale negative)
  // Correlation column & badge now removed – no longer needed.

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="text-green-700">Final Prediction Features</CardTitle>
        <CardDescription>
          Selected features for CO₂ emission prediction after VIF and correlation analysis
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Feature</TableHead>
              <TableHead className="text-center">VIF</TableHead>
              <TableHead>Description</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {sortedFeatures.map((feature, index) => (
              <TableRow key={feature.name} className="hover:bg-green-50">
                <motion.td
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.04 }}
                  className="font-medium p-3 align-middle"
                >
                  {formatFeatureName(feature.name)}
                </motion.td>
                <motion.td
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.04 }}
                  className="text-center p-3 align-middle"
                >
                  <Badge variant="outline" className="bg-gray-100">
                    {feature.vif.toFixed(2)}
                  </Badge>
                </motion.td>
                <motion.td
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.04 }}
                  className="p-3 align-middle"
                >
                  {feature.description}
                </motion.td>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
};

export default FeatureTable;

