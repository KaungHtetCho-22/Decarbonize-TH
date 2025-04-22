
import React from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Info } from "lucide-react";
import { correlationData, formatFeatureName, CorrelationPoint } from '@/utils/featureData';

interface CorrelationHeatmapProps {
  className?: string;
}

const CorrelationHeatmap: React.FC<CorrelationHeatmapProps> = ({ className }) => {
  // Sorted feature order with CO₂ last
  const co2Index = correlationData.map(d => d.x).lastIndexOf("co2");
  const featureNames = Array.from(new Set(correlationData.map(d => d.x).filter(x => x !== "co2")));
  featureNames.push("co2");

  // Enhanced red-green color scale for negative/positive
  const getColor = (value: number) => {
    if (value >= 0.8) return '#15803d';   // Strong positive - dark green
    if (value >= 0.5) return '#22c55e';   // Medium green
    if (value >= 0.2) return '#bbf7d0';   // Light green
    if (value > -0.2) return '#f3f4f6';   // Neutral/gray
    if (value > -0.5) return '#fdcfcf';   // Light red (visible)
    if (value > -0.8) return '#fb7185';   // Red-400
    return '#b91c1c';                     // Strong negative - dark red
  };

  // Multi-line ellipsis for long labels
  const wrapLabel = (str: string, maxLen: number) => {
    const words = str.split(' ');
    const lines = [];
    let curr = "";
    words.forEach(word => {
      if ((curr + " " + word).trim().length <= maxLen) {
        curr = (curr ? curr + " " : "") + word;
      } else {
        lines.push(curr);
        curr = word;
      }
    });
    if (curr) lines.push(curr);
    return lines;
  };

  return (
    <Card className={className + " overflow-x-auto bg-green-50/50 rounded-xl shadow-xl"}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-green-700">Feature Correlation Matrix</CardTitle>
            <CardDescription>
              Explore how different features relate to each other and to CO₂ emissions.
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-[440px] md:h-[600px] overflow-auto pb-6">
          <div className="min-w-[800px] min-h-[560px]">
            <div className="relative mb-2">
              <div className="absolute -top-5 left-0 text-xs text-gray-500 flex items-center gap-1">
                <Info size={16} className="inline mr-1" />
                Hover/tap for description and exact correlation
              </div>
            </div>
            <table className="w-full border-collapse bg-white rounded-xl shadow-lg overflow-hidden border border-green-100">
              <thead>
                <tr>
                  <th className="border p-2 bg-green-50 sticky left-0 z-10"></th>
                  {featureNames.map(name => (
                    <th
                      key={name}
                      className="border p-2 bg-green-50 text-xs font-bold whitespace-normal leading-tight"
                      style={{ minWidth: 72, maxWidth: 102, wordBreak: 'break-word' }}
                    >
                      <div className="max-w-[100px] break-words">
                        {wrapLabel(formatFeatureName(name), 16).map((line, idx) => (
                          <span className="block" key={idx}>{line}</span>
                        ))}
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {featureNames.map(rowName => (
                  <tr key={rowName}>
                    <th className="border p-2 bg-green-50 text-xs text-left w-32 sticky left-0 z-10 font-semibold leading-tight">
                      <div className="max-w-[106px] break-words">
                        {wrapLabel(formatFeatureName(rowName), 16).map((line, idx) => (
                          <span className="block" key={idx}>{line}</span>
                        ))}
                      </div>
                    </th>
                    {featureNames.map(colName => {
                      const point = correlationData.find(d => d.x === colName && d.y === rowName);
                      const value = point ? point.value : 0;
                      const isDiagonal = colName === rowName;
                      const cellVal = Math.abs(value) === 1 ? "1" : value.toFixed(2);
                      // Tooltip enhancement: show on hover/focus, popover inside cell for accessibility
                      return (
                        <td
                          key={`${rowName}-${colName}`}
                          className={`
                            border transition duration-100 p-0 relative group text-center align-middle font-semibold w-10 h-10
                            ${isDiagonal ? "opacity-80 bg-[#d1fae5]" : ""}
                            rounded-md
                          `}
                          style={{
                            backgroundColor: getColor(value),
                            borderRadius: isDiagonal ? 8 : 4,
                            border: isDiagonal ? "2px solid #a7f3d0" : undefined,
                            cursor: !isDiagonal ? 'pointer' : undefined,
                          }}
                        >
                          {!isDiagonal ? (
                            <div
                              className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 text-xs font-semibold select-none"
                            >
                              <span>{cellVal}</span>
                            </div>
                          ) : (
                            <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 text-xs opacity-60 font-normal">
                              —
                            </div>
                          )}
                          {/* Soft tooltip on hover for all but diagonal */}
                          {!isDiagonal && (
                            <div className="opacity-0 group-hover:opacity-100 pointer-events-none transition duration-200 absolute -top-10 left-1/2 -translate-x-1/2 bg-black text-white px-2 py-1 rounded text-xs z-20 whitespace-nowrap shadow-lg"
                              style={{ fontSize: 12, minWidth: 68, maxWidth: 220, wordBreak: "break-word" }}>
                              {formatFeatureName(rowName)} × {formatFeatureName(colName)}: {cellVal}
                            </div>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="flex flex-row flex-wrap justify-end items-center gap-4 mt-4 px-2">
              <div className="flex items-center gap-2">
                <span className="block h-4 w-8 rounded bg-[#15803d] border border-gray-300"></span>
                <span className="text-xs text-gray-700">Strong positive</span>
                <span className="block h-4 w-8 rounded bg-[#f3f4f6] border border-gray-300 mx-2"></span>
                <span className="text-xs text-gray-700">Neutral</span>
                <span className="block h-4 w-8 rounded bg-[#fb7185] border border-gray-300 mx-2"></span>
                <span className="text-xs text-gray-700">Strong negative</span>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default CorrelationHeatmap;
