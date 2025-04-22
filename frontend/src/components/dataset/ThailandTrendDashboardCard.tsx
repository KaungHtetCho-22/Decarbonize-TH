
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer } from "recharts";
import { thailandData } from "@/utils/thailandData";
import { useState } from "react";

const formatNumber = (value: number): string => {
  if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(2)}B`;
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(2)}K`;
  return value.toFixed(2);
};

const tinyChartConfig = {
  margin: { top: 6, right: 6, left: 6, bottom: 6 },
  height: 50,
  dot: false
};

// Encapsulate each tiny chart for reusability
const TinyLineChart = ({
  data,
  dataKey,
  stroke
}: {
  data: any[];
  dataKey: string;
  stroke: string;
}) => (
  <ResponsiveContainer width="100%" height={tinyChartConfig.height}>
    <LineChart data={data} margin={tinyChartConfig.margin}>
      <Line type="monotone" dataKey={dataKey} stroke={stroke} strokeWidth={2} dot={false} />
      <XAxis dataKey="year" hide />
      <YAxis hide domain={["auto", "auto"]} />
    </LineChart>
  </ResponsiveContainer>
);

const ThailandTrendDashboardCard = () => {
  const [tab, setTab] = useState("all");
  let filteredData = thailandData;
  if (tab === "recent") filteredData = thailandData.filter(d => d.year >= 1990);
  if (tab === "latest") filteredData = thailandData.filter(d => d.year >= 2010);

  // Find max value for y-labels
  const maxCO2 = Math.max(...filteredData.map(r => r.co2));
  const maxGDP = Math.max(...filteredData.map(r => r.gdp));
  const maxPop = Math.max(...filteredData.map(r => r.population));

  return (
    <Card className="h-full flex flex-col shadow-none bg-green-50/50 border border-green-100">
      <CardHeader>
        <CardTitle className="text-2xl text-green-700">Thailand Trend Dashboard</CardTitle>
        <CardDescription>
          Historical trends of CO₂ emissions, GDP, and population in Thailand from 1950 to present
        </CardDescription>
        <Tabs value={tab} onValueChange={setTab} className="w-full mt-2">
          <TabsList className="w-full bg-green-100 grid grid-cols-3">
            <TabsTrigger value="all">All Data (1950-Present)</TabsTrigger>
            <TabsTrigger value="recent">Recent (1990-Present)</TabsTrigger>
            <TabsTrigger value="latest">Latest (2010-Present)</TabsTrigger>
          </TabsList>
        </Tabs>
      </CardHeader>
      <CardContent className="flex-1 py-2 px-1">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 h-full">
          {/* CO2 */}
          <Card className="flex flex-col items-start justify-between h-36 min-h-[120px] border-green-200 px-4 py-3 rounded-xl">
            <span className="font-semibold text-green-800 text-lg">CO₂ Emissions<br /> <span className="font-normal text-base">(MtCO₂)</span></span>
            <div className="w-full h-[56px]"><TinyLineChart data={filteredData} dataKey="co2" stroke="#15803d" /></div>
            <div className="text-green-700 font-mono mt-1 text-xs">{maxCO2.toLocaleString(undefined, { maximumFractionDigits: 2 })}</div>
          </Card>
          {/* GDP */}
          <Card className="flex flex-col items-start justify-between h-36 min-h-[120px] border-green-200 px-4 py-3 rounded-xl">
            <span className="font-semibold text-green-800 text-lg">GDP <span className="font-normal text-base">(USD)</span></span>
            <div className="w-full h-[56px]"><TinyLineChart data={filteredData} dataKey="gdp" stroke="#0891b2" /></div>
            <div className="text-cyan-800 font-mono mt-1 text-xs">{formatNumber(maxGDP)} USD</div>
          </Card>
          {/* Population */}
          <Card className="flex flex-col items-start justify-between h-36 min-h-[120px] border-green-200 px-4 py-3 rounded-xl">
            <span className="font-semibold text-green-800 text-lg">Population</span>
            <div className="w-full h-[56px]"><TinyLineChart data={filteredData} dataKey="population" stroke="#7c3aed" /></div>
            <div className="text-purple-800 font-mono mt-1 text-xs">{formatNumber(maxPop)}</div>
          </Card>
        </div>
      </CardContent>
    </Card>
  );
};

export default ThailandTrendDashboardCard;
