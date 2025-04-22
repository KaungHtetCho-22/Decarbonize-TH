
import React from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { getFeatureInsights } from '@/utils/featureData';

interface FeatureInsightsProps {
  className?: string;
}

const FeatureInsights: React.FC<FeatureInsightsProps> = ({ className }) => {
  const insights = getFeatureInsights();

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="text-green-700">Key Insights</CardTitle>
        <CardDescription>
          Understanding the relationships between features and CO₂ emissions
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {insights.map((insight, index) => (
            <motion.div 
              key={index}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="border-l-4 border-green-500 pl-4 py-2"
            >
              <h3 className="font-medium text-green-700 text-lg">{insight.title}</h3>
              <p className="text-gray-600">{insight.description}</p>
            </motion.div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

export default FeatureInsights;
