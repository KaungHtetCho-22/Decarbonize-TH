import React, { useRef } from 'react';
import { motion, useScroll, useTransform } from "framer-motion";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie, 
  Scatter,
  ScatterChart,
  ZAxis
} from 'recharts';
import { Layers, Filter, ArrowRight, Database, LineChart } from "lucide-react";

const Features = () => {
  const scrollRef = useRef(null);
  const { scrollYProgress } = useScroll({
    target: scrollRef,
    offset: ["start start", "end start"]
  });
  
  const opacity = useTransform(scrollYProgress, [0, 0.5], [1, 0]);
  const scale = useTransform(scrollYProgress, [0, 0.5], [1, 0.95]);

  // Sample data
  const originalFeatures = [
    { name: "Energy Consumption (GWh)", description: "Total electricity usage in gigawatt-hours" },
    { name: "GDP (Billion Baht)", description: "Gross Domestic Product in billion Thai Baht" },
    { name: "Population (Million)", description: "Total population in millions" },
    { name: "Industrial Output (Index)", description: "Industrial production index" },
    { name: "Transport Activity (Million km)", description: "Total distance traveled by vehicles" },
    { name: "Agricultural Production (Ton)", description: "Agricultural output in tons" },
    { name: "Temperature (°C)", description: "Average annual temperature" },
    { name: "Rainfall (mm)", description: "asdfsafasfsadf"}
  ];

  const extractedFeatures = [
    { name: "Energy Intensity", description: "Energy consumption per unit of GDP" },
    { name: "Per Capita Emissions", description: "Carbon emissions divided by population" },
    { name: "Industrial Efficiency", description: "Emissions relative to industrial output" },
    { name: "Transport Efficiency", description: "Emissions per kilometer traveled" },
    { name: "Agricultural Intensity", description: "Emissions per ton of agricultural output" },
    { name: "Climate Factor", description: "Combined impact of temperature and rainfall" },
    { name: "Seasonal Variations", description: "Emission patterns across different seasons" },
    { name: "Urban Concentration", description: "Emissions concentration in urban areas" }
  ];

  // Correlation data
  const correlationData = [
    { name: "Energy Consumption", value: 0.92 },
    { name: "GDP", value: 0.85 },
    { name: "Population", value: 0.78 },
    { name: "Industrial Output", value: 0.81 },
    { name: "Transport Activity", value: 0.88 },
    { name: "Agricultural Production", value: 0.64 },
    { name: "Energy Intensity", value: 0.89 },
    { name: "Transport Efficiency", value: -0.72 }
  ];

  // Scatter plot data showing relationship between two features
  const scatterData = [
    { x: 120, y: 35, z: 10, name: 'Bangkok' },
    { x: 80, y: 28, z: 8, name: 'Chiang Mai' },
    { x: 95, y: 32, z: 9, name: 'Phuket' },
    { x: 60, y: 25, z: 7, name: 'Khon Kaen' },
    { x: 105, y: 34, z: 9.5, name: 'Pattaya' },
    { x: 70, y: 27, z: 7.5, name: 'Hat Yai' },
    { x: 90, y: 30, z: 8, name: 'Udon Thani' },
    { x: 75, y: 26, z: 6, name: 'Nakhon Ratchasima' },
    { x: 115, y: 33, z: 10, name: 'Rayong' }
  ];

  // PieChart data showing distribution of emission factors
  const emissionFactors = [
    { name: "Energy", value: 42 },
    { name: "Transport", value: 28 },
    { name: "Industry", value: 18 },
    { name: "Agriculture", value: 8 },
    { name: "Others", value: 4 }
  ];

  const COLORS = ['#22c55e', '#16a34a', '#84cc16', '#65a30d', '#4ade80'];

  return (
    <div className="min-h-screen pt-16">
      <motion.section
        ref={scrollRef}
        style={{ opacity, scale }}
        className="relative py-16 bg-gradient-to-b from-green-50 to-white overflow-hidden"
      >
        <div className="section-container text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="inline-block p-2 bg-green-100 rounded-full mb-4">
              <Layers className="h-8 w-8 text-green-600" />
            </div>
            <h1 className="text-4xl font-bold text-green-800 mb-6">Feature Analysis</h1>
            <p className="text-lg text-gray-700 max-w-3xl mx-auto mb-8">
              Explore the key features that drive carbon emissions in Thailand, from raw data inputs to 
              engineered features for improved prediction accuracy.
            </p>
          </motion.div>
        </div>
        
        {/* Animated background elements */}
        <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none">
          {[...Array(5)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute opacity-10"
              animate={{
                x: [Math.random() * 100, Math.random() * window.innerWidth],
                y: [Math.random() * 100, Math.random() * window.innerHeight],
              }}
              transition={{
                duration: 20 + Math.random() * 10,
                repeat: Infinity,
                repeatType: "reverse",
              }}
            >
              <svg width="60" height="60" viewBox="0 0 60 60" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M30 5L5 55L55 55L30 5Z" fill="#22c55e" />
              </svg>
            </motion.div>
          ))}
        </div>
      </motion.section>
      
      <section className="py-12">
        <div className="section-container">
          <Tabs defaultValue="original" className="w-full">
            <div className="flex justify-center mb-6">
              <TabsList className="grid grid-cols-2 w-[400px]">
                <TabsTrigger value="original" className="flex items-center">
                  <Database size={16} className="mr-2" />
                  Original Features
                </TabsTrigger>
                <TabsTrigger value="extracted" className="flex items-center">
                  <Filter size={16} className="mr-2" />
                  Extracted Features
                </TabsTrigger>
              </TabsList>
            </div>
            
            <TabsContent value="original">
              <Card>
                <CardHeader>
                  <CardTitle className="text-green-700">Original Dataset Features</CardTitle>
                  <CardDescription>
                    These are the raw features collected from various sources that form our base dataset.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableCaption>List of original features in the dataset</TableCaption>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Feature Name</TableHead>
                        <TableHead>Description</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {originalFeatures.map((feature) => (
                        <TableRow key={feature.name}>
                          <TableCell className="font-medium">{feature.name}</TableCell>
                          <TableCell>{feature.description}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="extracted">
              <Card>
                <CardHeader>
                  <CardTitle className="text-green-700">Engineered Features</CardTitle>
                  <CardDescription>
                    These features are extracted and transformed from the original dataset to improve model performance.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableCaption>List of engineered features used in models</TableCaption>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Feature Name</TableHead>
                        <TableHead>Description</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {extractedFeatures.map((feature) => (
                        <TableRow key={feature.name}>
                          <TableCell className="font-medium">{feature.name}</TableCell>
                          <TableCell>{feature.description}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </section>
      
      <section className="py-12 bg-green-50">
        <div className="section-container">
          <h2 className="section-title">Feature Correlations</h2>
          <p className="text-gray-700 mb-8">
            Understanding the relationship between different features and carbon emissions is essential
            for building accurate prediction models.
          </p>
          
          <div className="grid md:grid-cols-2 gap-8">
            <Card className="overflow-hidden">
              <CardHeader>
                <CardTitle className="text-green-700">Correlation with Carbon Emissions</CardTitle>
                <CardDescription>
                  Higher values indicate stronger correlation with emission levels
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[350px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={correlationData}
                      margin={{ top: 20, right: 30, left: 20, bottom: 70 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="name" 
                        angle={-45}
                        textAnchor="end"
                        height={70}
                      />
                      <YAxis domain={[-1, 1]} />
                      <Tooltip formatter={(value) => (typeof value === 'number' ? value.toFixed(2) : String(value))} />
                      <Legend />
                      <Bar dataKey="value" fill="#22c55e">
                        {correlationData.map((entry, index) => (
                          <Cell 
                            key={`cell-${index}`} 
                            fill={entry.value > 0 ? '#22c55e' : '#ef4444'} 
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
            
            <Card className="overflow-hidden">
              <CardHeader>
                <CardTitle className="text-green-700">Energy vs. Emissions by Region</CardTitle>
                <CardDescription>
                  Relationship between energy consumption and carbon emissions across regions
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[350px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart
                      margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                    >
                      <CartesianGrid />
                      <XAxis 
                        type="number" 
                        dataKey="x" 
                        name="Energy Consumption" 
                        unit="GWh"
                        label={{ 
                          value: 'Energy Consumption (GWh)', 
                          position: 'bottom', 
                          offset: 10 
                        }} 
                      />
                      <YAxis 
                        type="number" 
                        dataKey="y" 
                        name="Carbon Emissions" 
                        unit="MT"
                        label={{ 
                          value: 'Carbon Emissions (MT)', 
                          angle: -90, 
                          position: 'left' 
                        }}
                      />
                      <ZAxis type="number" dataKey="z" range={[60, 200]} name="Population" unit="M" />
                      <Tooltip 
                        cursor={{ strokeDasharray: '3 3' }}
                        formatter={(value, name, props) => {
                          if (name === 'Population') return [`${value}M`, name];
                          return [value, name];
                        }}
                        content={({ payload }) => {
                          if (!payload || payload.length === 0) return null;
                          return (
                            <div className="bg-white p-2 border border-gray-200 shadow-sm rounded-md">
                              <p className="font-medium text-gray-900">{payload[0].payload.name}</p>
                              <p className="text-sm text-gray-700">
                                Energy: {payload[0].value} GWh
                              </p>
                              <p className="text-sm text-gray-700">
                                Emissions: {payload[1].value} MT
                              </p>
                              <p className="text-sm text-gray-700">
                                Population: {payload[2].value}M
                              </p>
                            </div>
                          );
                        }}
                      />
                      <Scatter 
                        name="Regions" 
                        data={scatterData} 
                        fill="#15803d"
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>
      
      <section className="py-12">
        <div className="section-container">
          <h2 className="section-title">Emission Factors</h2>
          <p className="text-gray-700 mb-8">
            Analysis of the main contributing factors to Thailand's carbon emissions
          </p>
          
          <div className="grid md:grid-cols-2 gap-8">
            <Card>
              <CardHeader>
                <CardTitle className="text-green-700">Distribution of Emission Sources</CardTitle>
                <CardDescription>
                  Percentage contribution of different sectors to total carbon emissions
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[350px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={emissionFactors}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        outerRadius={100}
                        fill="#8884d8"
                        dataKey="value"
                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                      >
                        {emissionFactors.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip formatter={(value) => `${value}%`} />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
            
            <Card className="flex flex-col">
              <CardHeader>
                <CardTitle className="text-green-700">Feature Engineering Process</CardTitle>
                <CardDescription>
                  How raw data is transformed into meaningful predictive features
                </CardDescription>
              </CardHeader>
              <CardContent className="flex-grow">
                <div className="space-y-4">
                  {[
                    {
                      step: "Data Collection",
                      description: "Gathering raw data from government sources, industry reports, and environmental monitoring stations"
                    },
                    {
                      step: "Cleaning & Normalization",
                      description: "Handling missing values, outliers, and normalizing features to comparable scales"
                    },
                    {
                      step: "Feature Creation",
                      description: "Developing new derived features that capture complex relationships between original variables"
                    },
                    {
                      step: "Dimensionality Reduction",
                      description: "Using techniques like PCA to reduce redundancy while preserving information"
                    },
                    {
                      step: "Feature Selection",
                      description: "Selecting the most predictive features using statistical methods and domain knowledge"
                    }
                  ].map((item, index) => (
                    <motion.div 
                      key={index}
                      className="flex"
                      initial={{ opacity: 0, x: -20 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      viewport={{ once: true }}
                    >
                      <div className="mr-4 flex flex-col items-center">
                        <div className="w-8 h-8 rounded-full bg-green-600 flex items-center justify-center text-white font-medium">
                          {index + 1}
                        </div>
                        {index < 4 && <div className="w-0.5 h-full bg-green-200 my-1"></div>}
                      </div>
                      <div className="mb-4">
                        <h4 className="font-medium text-green-700">{item.step}</h4>
                        <p className="text-gray-600">{item.description}</p>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>
      
      <section className="py-12 bg-gradient-to-br from-green-700 to-green-900 text-white">
        <div className="section-container">
          <div className="text-center">
            <h2 className="text-3xl font-bold mb-4">Learn More About Our Data</h2>
            <p className="max-w-2xl mx-auto mb-8 text-green-100">
              See how we've transformed raw data into a clean dataset ready for machine learning models.
            </p>
            <motion.div
              className="inline-block"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <a 
                href="/dataset" 
                className="inline-flex items-center bg-white text-green-700 px-6 py-3 rounded-md font-medium hover:bg-green-50 transition-colors"
              >
                Explore Dataset
                <ArrowRight className="ml-2 h-5 w-5" />
              </a>
            </motion.div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Features;
