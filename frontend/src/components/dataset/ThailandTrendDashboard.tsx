
import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Legend,
  ReferenceLine
} from 'recharts';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { thailandData } from "@/utils/thailandData";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";

const formatNumber = (value: number): string => {
  if (value >= 1_000_000_000) {
    return `${(value / 1_000_000_000).toFixed(2)}B`;
  } else if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(2)}M`;
  } else if (value >= 1_000) {
    return `${(value / 1_000).toFixed(2)}K`;
  }
  return value.toFixed(2);
};

const ThailandTrendDashboard = () => {
  const [activeTab, setActiveTab] = useState("all");
  
  const getFilteredData = () => {
    if (activeTab === "all") return thailandData;
    if (activeTab === "recent") return thailandData.filter(d => d.year >= 1990);
    return thailandData.filter(d => d.year >= 2010);
  };

  return (
    <Card className="mb-8">
      <CardHeader>
        <CardTitle className="text-2xl text-green-700">Thailand Trend Dashboard</CardTitle>
        <CardDescription>
          Historical trends of CO₂ emissions, GDP, and population in Thailand from 1950 to present
        </CardDescription>
        <Tabs 
          value={activeTab} 
          onValueChange={setActiveTab}
          className="w-full"
        >
          <TabsList className="grid w-full sm:w-auto grid-cols-3 sm:flex">
            <TabsTrigger value="all">All Data (1950-Present)</TabsTrigger>
            <TabsTrigger value="recent">Recent (1990-Present)</TabsTrigger>
            <TabsTrigger value="latest">Latest (2010-Present)</TabsTrigger>
          </TabsList>
        </Tabs>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* CO2 Emissions Chart */}
          <Card className="bg-white/50 shadow-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg text-green-700">CO₂ Emissions (MtCO₂)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-[250px]">
                <ChartContainer
                  config={{
                    co2: {
                      label: "CO₂ Emissions",
                      color: "#15803d"
                    }
                  }}
                >
                  <LineChart data={getFilteredData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="year" 
                      tickFormatter={(value) => value % 10 === 0 ? value.toString() : ''}
                    />
                    <YAxis tickFormatter={(value) => formatNumber(Number(value))} />
                    <ChartTooltip
                      content={({ active, payload, label }) => {
                        if (active && payload && payload.length) {
                          return (
                            <div className="bg-white p-3 border border-gray-200 rounded-md shadow-md">
                              <p className="font-medium">Year: {label}</p>
                              <p className="text-green-700">
                                CO₂: {Number(payload[0].value).toFixed(3)} MtCO₂
                              </p>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="co2" 
                      stroke="#15803d" 
                      strokeWidth={2} 
                      dot={{ r: 1 }}
                      activeDot={{ r: 5 }}
                    />
                  </LineChart>
                </ChartContainer>
              </div>
            </CardContent>
          </Card>

          {/* GDP Chart */}
          <Card className="bg-white/50 shadow-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg text-green-700">GDP (USD)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-[250px]">
                <ChartContainer
                  config={{
                    gdp: {
                      label: "GDP",
                      color: "#0891b2"
                    }
                  }}
                >
                  <LineChart data={getFilteredData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="year" 
                      tickFormatter={(value) => value % 10 === 0 ? value.toString() : ''}
                    />
                    <YAxis tickFormatter={(value) => formatNumber(Number(value))} />
                    <ChartTooltip
                      content={({ active, payload, label }) => {
                        if (active && payload && payload.length) {
                          return (
                            <div className="bg-white p-3 border border-gray-200 rounded-md shadow-md">
                              <p className="font-medium">Year: {label}</p>
                              <p className="text-cyan-700">
                                GDP: {formatNumber(Number(payload[0].value))} USD
                              </p>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="gdp" 
                      stroke="#0891b2" 
                      strokeWidth={2} 
                      dot={{ r: 1 }}
                      activeDot={{ r: 5 }}
                    />
                  </LineChart>
                </ChartContainer>
              </div>
            </CardContent>
          </Card>

          {/* Population Chart */}
          <Card className="bg-white/50 shadow-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg text-green-700">Population</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-[250px]">
                <ChartContainer
                  config={{
                    population: {
                      label: "Population",
                      color: "#7c3aed"
                    }
                  }}
                >
                  <LineChart data={getFilteredData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="year" 
                      tickFormatter={(value) => value % 10 === 0 ? value.toString() : ''}
                    />
                    <YAxis tickFormatter={(value) => formatNumber(Number(value))} />
                    <ChartTooltip
                      content={({ active, payload, label }) => {
                        if (active && payload && payload.length) {
                          return (
                            <div className="bg-white p-3 border border-gray-200 rounded-md shadow-md">
                              <p className="font-medium">Year: {label}</p>
                              <p className="text-purple-700">
                                Population: {formatNumber(Number(payload[0].value))}
                              </p>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="population" 
                      stroke="#7c3aed" 
                      strokeWidth={2} 
                      dot={{ r: 1 }}
                      activeDot={{ r: 5 }}
                    />
                  </LineChart>
                </ChartContainer>
              </div>
            </CardContent>
          </Card>
        </div>
      </CardContent>
    </Card>
  );
};

export default ThailandTrendDashboard;
