
import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Cell
} from 'recharts';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { featureData, formatFeatureName } from '@/utils/featureData';

// Updated: handle large feature names and better fit visually.
const MAX_LABEL_LINE_LEN = 19; // Shorter for improved wrapping
const BAR_HEIGHT = 28; // Height per item for more room (will pass to chart height)

interface FeatureCorrelationChartProps {
  className?: string;
}

const FeatureCorrelationChart: React.FC<FeatureCorrelationChartProps> = ({ className }) => {
  // Prepare data
  const chartData = featureData
    .map(feature => ({
      name: feature.name, // Use full name
      displayName: formatFeatureName(feature.name), // Optionally, but stick to full name in axis
      value: feature.correlation
    }))
    .sort((a, b) => b.value - a.value);

  // Color scale for correlation range
  const getBarColor = (correlation: number) => {
    // Vivid greens/blues for positive, reds for negative, grays for neutral
    if (correlation >= 0.8) return '#15803d';
    if (correlation >= 0.5) return '#22c55e';
    if (correlation >= 0.2) return '#bbf7d0';
    if (correlation > -0.2) return '#f3f4f6';
    if (correlation > -0.5) return '#fca5a5';
    if (correlation > -0.8) return '#fb7185';
    return '#b91c1c';
  };

  // More robust Y-axis label rendering for long feature names
  const renderYAxisTick = (props: any) => {
    const { x, y, payload, width, height } = props;
    // Word wrap for multi-line labels
    const fullLabel: string = payload.value;
    const words = fullLabel.split(/[\s_]/);
    let lines: string[] = [];
    let curr = "";
    words.forEach(word => {
      if ((curr + ' ' + word).trim().length <= MAX_LABEL_LINE_LEN) {
        curr = (curr ? curr + " " : "") + word;
      } else {
        if (curr) lines.push(curr);
        curr = word;
      }
    });
    if (curr) lines.push(curr);
    // Ellipse if still too long, max 3 lines
    if (lines.length > 3) {
      lines = [...lines.slice(0,2), fullLabel.slice(0,28) + "…"];
    }
    return (
      <g>
        {lines.map((line, idx) => (
          <text
            key={idx}
            x={x - 6}
            y={y + idx * 14 - ((lines.length - 1) * 8)}
            textAnchor="end"
            width={146}
            fill="#065f46"
            style={{
              fontSize: '13px',
              fontWeight: 700,
              lineHeight: 1.1,
              textShadow: "0 1px 0 #fff, 0 0px 1px #eee"
            }}
            dominantBaseline="middle"
          >
            {line}
          </text>
        ))}
      </g>
    );
  };

  // Dynamic height for perfect bar fit
  const chartHeight = Math.max(chartData.length * BAR_HEIGHT + 40, 360);

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="text-green-700">Feature Correlation with CO₂</CardTitle>
        <CardDescription>
          <span>
            Strength/Direction of correlation for each feature with CO₂ emissions.
          </span>
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="w-full" style={{ height: chartHeight }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 8, right: 44, left: 200, bottom: 8 }}
              barCategoryGap="16%"
            >
              <CartesianGrid strokeDasharray="3 5" horizontal vertical={false} />
              <XAxis
                type="number"
                domain={[-1, 1]}
                tickCount={9}
                ticks={[-1, -0.8, -0.5, -0.2, 0, 0.2, 0.5, 0.8, 1]}
                axisLine={false}
                tickLine={false}
                fontSize={13}
                stroke="#aaaaaa"
                tickFormatter={v => v.toFixed(1)}
                style={{ fontWeight: 500 }}
              />
              <YAxis
                dataKey="name"
                type="category"
                width={196}
                tick={renderYAxisTick}
                tickLine={false}
                axisLine={false}
                interval={0}
              />
              <Tooltip
                cursor={{ fill: '#eee', opacity: 0.38 }}
                formatter={(value: number) => [`${value.toFixed(2)}`, 'Correlation']}
                labelFormatter={(label) => `Feature: ${label}`}
                wrapperClassName="!rounded-md !text-xs !shadow-2xl"
              />
              <Bar
                dataKey="value"
                minPointSize={7}
                radius={[6, 6, 6, 6]}
                isAnimationActive
                barSize={18}
              >
                {chartData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={getBarColor(entry.value)}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};

export default FeatureCorrelationChart;
