import { useState } from "react";
import { motion } from "framer-motion";
import { 
  Tabs, 
  TabsContent, 
  TabsList, 
  TabsTrigger 
} from "@/components/ui/tabs";
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
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  Cell,
  Legend
} from 'recharts';
import { Database, Filter, Brush, AlertCircle, Info, Check, ArrowRight } from "lucide-react";

const Dataset = () => {
  const [showPreprocessing, setShowPreprocessing] = useState(false);

  // Sample data for the original dataset
  const originalData = [
    { year: 2012, energy: 174.2, emissions: 232.5, gdp: 12.7, population: 67.8, industry: 98.3, transport: 78.6, missing: true },
    { year: 2013, energy: 180.1, emissions: 241.3, gdp: 13.2, population: 68.3, industry: 102.1, transport: 82.4, missing: false },
    { year: 2014, energy: 183.7, emissions: 249.8, gdp: 13.4, population: 68.7, industry: 103.9, transport: 85.7, missing: false },
    { year: 2015, energy: 189.5, emissions: 259.3, gdp: 13.9, population: 69.1, industry: 106.2, transport: 88.9, missing: false },
    { year: 2016, energy: 195.2, emissions: 267.1, gdp: 14.2, population: 69.5, industry: 109.7, transport: 91.2, missing: true },
    { year: 2017, energy: 201.3, emissions: 274.6, gdp: 14.6, population: 69.8, industry: 112.3, transport: 95.8, missing: false },
    { year: 2018, energy: 208.7, emissions: 283.4, gdp: 15.1, population: 70.2, industry: 115.1, transport: 98.3, missing: false },
    { year: 2019, energy: 215.9, emissions: 291.7, gdp: 15.5, population: 70.5, industry: 117.8, transport: 101.4, missing: true },
    { year: 2020, energy: 198.2, emissions: 267.8, gdp: 14.8, population: 70.7, industry: 109.6, transport: 87.5, missing: false },
    { year: 2021, energy: 209.5, emissions: 281.9, gdp: 15.2, population: 71.0, industry: 114.3, transport: 92.6, missing: false },
  ];
  
  // Sample data for the cleaned dataset
  const cleanedData = [
    { year: 2012, energy: 174.2, emissions: 232.5, gdp: 12.7, population: 67.8, industry: 98.3, transport: 78.6, energyIntensity: 1.37, perCapita: 3.43 },
    { year: 2013, energy: 180.1, emissions: 241.3, gdp: 13.2, population: 68.3, industry: 102.1, transport: 82.4, energyIntensity: 1.36, perCapita: 3.53 },
    { year: 2014, energy: 183.7, emissions: 249.8, gdp: 13.4, population: 68.7, industry: 103.9, transport: 85.7, energyIntensity: 1.37, perCapita: 3.64 },
    { year: 2015, energy: 189.5, emissions: 259.3, gdp: 13.9, population: 69.1, industry: 106.2, transport: 88.9, energyIntensity: 1.36, perCapita: 3.75 },
    { year: 2016, energy: 195.2, emissions: 267.1, gdp: 14.2, population: 69.5, industry: 109.7, transport: 91.2, energyIntensity: 1.37, perCapita: 3.84 },
    { year: 2017, energy: 201.3, emissions: 274.6, gdp: 14.6, population: 69.8, industry: 112.3, transport: 95.8, energyIntensity: 1.38, perCapita: 3.94 },
    { year: 2018, energy: 208.7, emissions: 283.4, gdp: 15.1, population: 70.2, industry: 115.1, transport: 98.3, energyIntensity: 1.38, perCapita: 4.04 },
    { year: 2019, energy: 215.9, emissions: 291.7, gdp: 15.5, population: 70.5, industry: 117.8, transport: 101.4, energyIntensity: 1.39, perCapita: 4.14 },
    { year: 2020, energy: 198.2, emissions: 267.8, gdp: 14.8, population: 70.7, industry: 109.6, transport: 87.5, energyIntensity: 1.34, perCapita: 3.79 },
    { year: 2021, energy: 209.5, emissions: 281.9, gdp: 15.2, population: 71.0, industry: 114.3, transport: 92.6, energyIntensity: 1.38, perCapita: 3.97 },
  ];

  // Missing values statistics
  const missingStats = [
    { feature: "Energy Consumption", percentage: 8.2 },
    { feature: "Carbon Emissions", percentage: 5.7 },
    { feature: "GDP", percentage: 2.1 },
    { feature: "Population", percentage: 0.5 },
    { feature: "Industrial Output", percentage: 9.3 },
    { feature: "Transport Activity", percentage: 7.6 },
    { feature: "Agricultural Production", percentage: 12.4 },
    { feature: "Temperature", percentage: 3.2 },
  ];

  // Data quality issues
  const dataQualityIssues = [
    { issue: "Missing Values", count: 214, description: "Data points that were not recorded or lost during collection" },
    { issue: "Outliers", count: 47, description: "Extreme values that significantly deviate from other observations" },
    { issue: "Inconsistent Units", count: 12, description: "Different measurement units used for the same feature" },
    { issue: "Duplicate Records", count: 28, description: "Identical data points recorded multiple times" },
    { issue: "Invalid Dates", count: 5, description: "Date entries that are impossible or outside the study timeframe" }
  ];
  
  // Data cleaning steps
  const cleaningSteps = [
    {
      title: "Missing Value Imputation",
      description: "We used multiple imputation techniques based on the nature of each feature:",
      details: [
        "Time series interpolation for energy and emissions data",
        "Mean substitution for demographic features with low variance",
        "KNN imputation for features with strong correlations",
        "Some records with excessive missing values (>30%) were removed"
      ]
    },
    {
      title: "Outlier Detection & Treatment",
      description: "Outliers were identified and handled using statistical methods:",
      details: [
        "Z-score and IQR methods to detect outliers",
        "Domain-specific thresholds based on expert knowledge",
        "Winsorization to cap extreme values at the 5th and 95th percentiles",
        "Log transformation for heavily skewed distributions"
      ]
    },
    {
      title: "Feature Engineering",
      description: "We created new features to better capture patterns in the data:",
      details: [
        "Energy intensity ratio (Energy consumption/GDP)",
        "Per capita emissions (Emissions/Population)",
        "Sectoral efficiency metrics for industry and transport",
        "Seasonal indicators and climate factors"
      ]
    },
    {
      title: "Data Integration",
      description: "Multiple data sources were combined to create a comprehensive dataset:",
      details: [
        "Aligning time periods across different sources",
        "Standardizing geographic regions for consistent analysis",
        "Converting units to ensure uniformity across features",
        "Resolving conflicts between overlapping data sources"
      ]
    }
  ];

  // Dataset statistics
  const datasetStats = {
    original: {
      records: 2450,
      timespan: "2002-2021",
      features: 32,
      sources: 8,
      completeness: "87.3%"
    },
    cleaned: {
      records: 2384,
      timespan: "2002-2021",
      features: 48,
      sources: 8,
      completeness: "99.8%"
    }
  };

  // Chart data for emissions trend
  const emissionsTrend = [
    { year: "2012", value: 232.5 },
    { year: "2013", value: 241.3 },
    { year: "2014", value: 249.8 },
    { year: "2015", value: 259.3 },
    { year: "2016", value: 267.1 },
    { year: "2017", value: 274.6 },
    { year: "2018", value: 283.4 },
    { year: "2019", value: 291.7 },
    { year: "2020", value: 267.8 },
    { year: "2021", value: 281.9 }
  ];

  // Data sources
  const dataSources = [
    { name: "Thailand Office of Natural Resources and Environmental Policy and Planning", type: "Government", features: "Emissions data, climate policies", years: "2002-2021" },
    { name: "Electricity Generating Authority of Thailand", type: "Government", features: "Energy production and consumption", years: "2002-2021" },
    { name: "National Statistical Office of Thailand", type: "Government", features: "Demographic data, economic indicators", years: "2002-2021" },
    { name: "Ministry of Industry", type: "Government", features: "Industrial production indices", years: "2004-2021" },
    { name: "Department of Land Transport", type: "Government", features: "Transportation statistics", years: "2005-2021" },
    { name: "Thai Meteorological Department", type: "Government", features: "Climate data, temperature, rainfall", years: "2002-2021" },
    { name: "World Bank Open Data", type: "International", features: "Economic indicators, comparative statistics", years: "2002-2021" },
    { name: "Global Carbon Project", type: "Research", features: "Carbon emission estimates, verification data", years: "2002-2020" }
  ];

  const COLORS = ['#22c55e', '#16a34a', '#4ade80', '#86efac', '#bbf7d0'];

  return (
    <div className="min-h-screen pt-16">
      <motion.section 
        className="relative py-16 bg-gradient-to-b from-green-50 to-white overflow-hidden"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6 }}
      >
        <div className="section-container text-center">
          <div className="inline-block p-2 bg-green-100 rounded-full mb-4">
            <Database className="h-8 w-8 text-green-600" />
          </div>
          <h1 className="text-4xl font-bold text-green-800 mb-6">Dataset Overview</h1>
          <p className="text-lg text-gray-700 max-w-3xl mx-auto mb-8">
            Explore the data that powers our carbon emission predictions, from raw collection to 
            cleaned and processed datasets ready for modeling.
          </p>
        </div>
        
        <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
          {[...Array(20)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute bg-green-200 opacity-20 rounded-full"
              style={{
                height: 10 + Math.random() * 20,
                width: 10 + Math.random() * 20,
                top: Math.random() * 100 + '%',
                left: Math.random() * 100 + '%',
              }}
              animate={{
                y: [0, -10, 0],
                opacity: [0.1, 0.3, 0.1],
              }}
              transition={{
                duration: 2 + Math.random() * 3,
                repeat: Infinity,
                delay: Math.random() * 5,
              }}
            />
          ))}
        </div>
      </motion.section>
      
      <section className="py-12">
        <div className="section-container">
          <div className="grid md:grid-cols-2 gap-8 mb-12">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
              viewport={{ once: true }}
            >
              <Card>
                <CardHeader>
                  <CardTitle className="text-green-700">Dataset Statistics</CardTitle>
                  <CardDescription>Comparing original and cleaned datasets</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-6">
                    <div>
                      <h3 className="text-sm font-medium text-muted-foreground mb-3">Original Dataset</h3>
                      <ul className="space-y-2">
                        <li className="flex justify-between">
                          <span className="text-gray-600">Records:</span>
                          <span className="font-medium">{datasetStats.original.records}</span>
                        </li>
                        <li className="flex justify-between">
                          <span className="text-gray-600">Time Span:</span>
                          <span className="font-medium">{datasetStats.original.timespan}</span>
                        </li>
                        <li className="flex justify-between">
                          <span className="text-gray-600">Features:</span>
                          <span className="font-medium">{datasetStats.original.features}</span>
                        </li>
                        <li className="flex justify-between">
                          <span className="text-gray-600">Data Sources:</span>
                          <span className="font-medium">{datasetStats.original.sources}</span>
                        </li>
                        <li className="flex justify-between">
                          <span className="text-gray-600">Completeness:</span>
                          <span className="font-medium">{datasetStats.original.completeness}</span>
                        </li>
                      </ul>
                    </div>
                    <div>
                      <h3 className="text-sm font-medium text-muted-foreground mb-3">Cleaned Dataset</h3>
                      <ul className="space-y-2">
                        <li className="flex justify-between">
                          <span className="text-gray-600">Records:</span>
                          <span className="font-medium">{datasetStats.cleaned.records}</span>
                        </li>
                        <li className="flex justify-between">
                          <span className="text-gray-600">Time Span:</span>
                          <span className="font-medium">{datasetStats.cleaned.timespan}</span>
                        </li>
                        <li className="flex justify-between">
                          <span className="text-gray-600">Features:</span>
                          <span className="font-medium">{datasetStats.cleaned.features}</span>
                        </li>
                        <li className="flex justify-between">
                          <span className="text-gray-600">Data Sources:</span>
                          <span className="font-medium">{datasetStats.cleaned.sources}</span>
                        </li>
                        <li className="flex justify-between">
                          <span className="text-gray-600">Completeness:</span>
                          <span className="font-medium text-green-600">{datasetStats.cleaned.completeness}</span>
                        </li>
                      </ul>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              viewport={{ once: true }}
            >
              <Card>
                <CardHeader>
                  <CardTitle className="text-green-700">Carbon Emissions Trend</CardTitle>
                  <CardDescription>20-year annual carbon emissions in Thailand (MT CO₂)</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[250px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart
                        data={emissionsTrend}
                        margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="year" />
                        <YAxis />
                        <Tooltip 
                          formatter={(value) => [`${value} MT`, 'Emissions']}
                          labelFormatter={(label) => `Year: ${label}`}
                        />
                        <Area 
                          type="monotone" 
                          dataKey="value" 
                          stroke="#15803d" 
                          fill="#86efac" 
                          fillOpacity={0.6}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="flex items-center justify-center mt-4 text-sm text-gray-500">
                    <Info size={16} className="mr-2 text-green-600" />
                    Note the decrease in 2020 due to COVID-19 pandemic restrictions
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </div>
          
          <Tabs defaultValue="original">
            <div className="flex justify-center mb-6">
              <TabsList className="grid grid-cols-2 w-[400px]">
                <TabsTrigger value="original" className="flex items-center">
                  <Database size={16} className="mr-2" />
                  Original Dataset
                </TabsTrigger>
                <TabsTrigger value="cleaned" className="flex items-center">
                  <Brush size={16} className="mr-2" />
                  Cleaned Dataset
                </TabsTrigger>
              </TabsList>
            </div>
            
            <TabsContent value="original">
              <Card>
                <CardHeader>
                  <CardTitle className="text-green-700">Original Dataset Sample</CardTitle>
                  <CardDescription>
                    Raw data before cleaning and preprocessing (sample rows shown)
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <Table>
                      <TableCaption>Note: Cells with * indicate missing values in the original dataset</TableCaption>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Year</TableHead>
                          <TableHead>Energy (TWh)</TableHead>
                          <TableHead>Emissions (MT)</TableHead>
                          <TableHead>GDP (Trillion ฿)</TableHead>
                          <TableHead>Population (M)</TableHead>
                          <TableHead>Industry Index</TableHead>
                          <TableHead>Transport (B km)</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {originalData.map((row, index) => (
                          <TableRow key={index} className={row.missing ? "bg-red-50" : ""}>
                            <TableCell>{row.year}</TableCell>
                            <TableCell>{row.energy}</TableCell>
                            <TableCell>{row.emissions}</TableCell>
                            <TableCell>{row.gdp}</TableCell>
                            <TableCell>{row.population}</TableCell>
                            <TableCell>
                              {row.missing && index === 0 ? <span className="text-red-500">*</span> : row.industry}
                            </TableCell>
                            <TableCell>
                              {row.missing && index === 5 ? <span className="text-red-500">*</span> : row.transport}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                  
                  <div className="mt-8">
                    <h3 className="text-lg font-medium text-green-700 mb-4">Data Quality Issues</h3>
                    
                    <div className="grid md:grid-cols-2 gap-6">
                      <div>
                        <h4 className="text-sm font-medium text-gray-700 mb-3">Missing Values by Feature</h4>
                        <div className="h-[250px]">
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart
                              data={missingStats}
                              layout="vertical"
                              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                            >
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis type="number" domain={[0, 15]} unit="%" />
                              <YAxis dataKey="feature" type="category" width={150} />
                              <Tooltip formatter={(value) => [`${value}%`, 'Missing']} />
                              <Bar dataKey="percentage" fill="#ef4444" />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                      
                      <div>
                        <h4 className="text-sm font-medium text-gray-700 mb-3">Data Quality Issues</h4>
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>Issue Type</TableHead>
                              <TableHead>Count</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {dataQualityIssues.map((issue, index) => (
                              <TableRow key={index}>
                                <TableCell>
                                  <div className="flex items-center">
                                    <AlertCircle size={16} className="mr-2 text-amber-500" />
                                    <span>{issue.issue}</span>
                                  </div>
                                </TableCell>
                                <TableCell>{issue.count}</TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </div>
                    </div>
                    
                    <div className="mt-6">
                      <Button
                        onClick={() => setShowPreprocessing(true)}
                        variant="outline"
                        className="text-green-700 border-green-600"
                      >
                        <Filter className="mr-2 h-4 w-4" />
                        Show Data Preprocessing Steps
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="cleaned">
              <Card>
                <CardHeader>
                  <CardTitle className="text-green-700">Cleaned Dataset Sample</CardTitle>
                  <CardDescription>
                    Processed data after cleaning, imputation and feature engineering
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <Table>
                      <TableCaption>Complete dataset with engineered features</TableCaption>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Year</TableHead>
                          <TableHead>Energy (TWh)</TableHead>
                          <TableHead>Emissions (MT)</TableHead>
                          <TableHead>GDP (T฿)</TableHead>
                          <TableHead>Population (M)</TableHead>
                          <TableHead>Energy Intensity</TableHead>
                          <TableHead>Per Capita (t)</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {cleanedData.map((row, index) => (
                          <TableRow key={index}>
                            <TableCell>{row.year}</TableCell>
                            <TableCell>{row.energy}</TableCell>
                            <TableCell>{row.emissions}</TableCell>
                            <TableCell>{row.gdp}</TableCell>
                            <TableCell>{row.population}</TableCell>
                            <TableCell>{row.energyIntensity}</TableCell>
                            <TableCell>{row.perCapita}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                  
                  <div className="mt-8">
                    <h3 className="text-lg font-medium text-green-700 mb-4">Dataset Insights</h3>
                    
                    <div className="grid md:grid-cols-2 gap-6">
                      <div>
                        <h4 className="text-sm font-medium text-gray-700 mb-3">Feature Relationships</h4>
                        <div className="h-[250px]">
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart
                              data={cleanedData}
                              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                            >
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="year" />
                              <YAxis yAxisId="left" />
                              <YAxis yAxisId="right" orientation="right" />
                              <Tooltip />
                              <Legend />
                              <Line
                                yAxisId="left"
                                type="monotone"
                                dataKey="emissions"
                                name="Emissions (MT)"
                                stroke="#15803d"
                                activeDot={{ r: 8 }}
                              />
                              <Line
                                yAxisId="right"
                                type="monotone"
                                dataKey="energy"
                                name="Energy (TWh)"
                                stroke="#0284c7"
                                activeDot={{ r: 8 }}
                              />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                        <div className="mt-4 text-sm text-gray-600">
                          <p>Strong correlation between energy consumption and carbon emissions, with slight efficiency improvements over time.</p>
                        </div>
                      </div>
                      
                      <div>
                        <h4 className="text-sm font-medium text-gray-700 mb-3">Per Capita Emissions Trend</h4>
                        <div className="h-[250px]">
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart
                              data={cleanedData}
                              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                            >
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="year" />
                              <YAxis domain={[2, 5]} />
                              <Tooltip formatter={(value) => [`${value} tonnes`, 'Per Capita']} />
                              <Bar dataKey="perCapita" name="Per Capita Emissions (tonnes)">
                                {cleanedData.map((entry, index) => (
                                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                ))}
                              </Bar>
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                        <div className="mt-4 text-sm text-gray-600">
                          <p>Per capita emissions have been generally increasing, with a notable decrease during the COVID-19 pandemic in 2020.</p>
                        </div>
                      </div>
                    </div>
                    
                    {showPreprocessing ? (
                      <div className="mt-6">
                        <Button
                          onClick={() => setShowPreprocessing(false)}
                          variant="outline"
                          className="text-green-700 border-green-600"
                        >
                          <Check className="mr-2 h-4 w-4" />
                          Hide Data Preprocessing Steps
                        </Button>
                      </div>
                    ) : (
                      <div className="mt-6">
                        <Button
                          onClick={() => setShowPreprocessing(true)}
                          variant="outline"
                          className="text-green-700 border-green-600"
                        >
                          <Filter className="mr-2 h-4 w-4" />
                          Show Data Preprocessing Steps
                        </Button>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
          
          {showPreprocessing && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="mt-8"
            >
              <Card>
                <CardHeader>
                  <CardTitle className="text-green-700">Data Preprocessing and Cleaning</CardTitle>
                  <CardDescription>
                    Steps taken to transform the raw dataset into a clean, analysis-ready format
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Accordion type="single" collapsible className="w-full">
                    {cleaningSteps.map((step, index) => (
                      <AccordionItem key={index} value={`step-${index}`}>
                        <AccordionTrigger className="text-green-700 hover:text-green-800">
                          {step.title}
                        </AccordionTrigger>
                        <AccordionContent>
                          <p className="mb-3 text-gray-700">{step.description}</p>
                          <ul className="space-y-2 pl-5 list-disc text-gray-600">
                            {step.details.map((detail, idx) => (
                              <li key={idx}>{detail}</li>
                            ))}
                          </ul>
                        </AccordionContent>
                      </AccordionItem>
                    ))}
                  </Accordion>
                </CardContent>
              </Card>
            </motion.div>
          )}
        </div>
      </section>
      
      <section className="py-12 bg-green-50">
        <div className="section-container">
          <h2 className="section-title">Data Sources</h2>
          <p className="text-gray-700 mb-8">
            The dataset was compiled from multiple credible sources to ensure comprehensive coverage
            and high data quality.
          </p>
          
          <div className="overflow-x-auto">
            <Table>
              <TableCaption>Data sources used in the Decarbonize-TH project</TableCaption>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[300px]">Source</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Features Provided</TableHead>
                  <TableHead>Years Covered</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {dataSources.map((source, index) => (
                  <TableRow key={index}>
                    <TableCell className="font-medium">{source.name}</TableCell>
                    <TableCell>{source.type}</TableCell>
                    <TableCell>{source.features}</TableCell>
                    <TableCell>{source.years}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </div>
      </section>
      
      <section className="py-12 bg-gradient-to-br from-green-700 to-green-900 text-white">
        <div className="section-container">
          <div className="text-center">
            <h2 className="text-3xl font-bold mb-4">Ready to Explore Our Models?</h2>
            <p className="max-w-2xl mx-auto mb-8 text-green-100">
              See how we've used this dataset to build predictive models for carbon emissions in Thailand.
            </p>
            <motion.div
              className="inline-block"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <a 
                href="/models" 
                className="inline-flex items-center bg-white text-green-700 px-6 py-3 rounded-md font-medium hover:bg-green-50 transition-colors"
              >
                Explore Models
                <ArrowRight className="ml-2 h-5 w-5" />
              </a>
            </motion.div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Dataset;
