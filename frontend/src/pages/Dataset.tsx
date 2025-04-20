
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
import { Database, Filter, CircleCheck, AlertCircle, Info, FileText } from "lucide-react";

const Dataset = () => {
  const [showPreprocessing, setShowPreprocessing] = useState(false);

  // Sample data for visualization
  const cleanedData = [
    { year: 1950, co2: 25.3, energy: 42.1, population: 20.1 },
    { year: 1960, co2: 45.7, energy: 68.4, population: 27.3 },
    { year: 1970, co2: 78.2, energy: 112.6, population: 36.9 },
    { year: 1980, co2: 134.5, energy: 187.3, population: 47.4 },
    { year: 1990, co2: 205.8, energy: 278.9, population: 56.6 },
    { year: 2000, co2: 298.4, energy: 389.5, population: 62.7 },
    { year: 2010, co2: 367.2, energy: 487.6, population: 67.2 },
    { year: 2020, co2: 412.6, energy: 534.8, population: 69.8 }
  ];

  // Data cleaning steps based on actual notebook logic
  const cleaningSteps = [
    {
      title: "Initial Data Loading & Filtering",
      description: "Source: Our World in Data CO₂ Dataset",
      details: [
        "Filter records from 1950 onward only",
        "Remove region-level aggregates by filtering to valid ISO-3 country codes",
        "Select relevant features for analysis"
      ]
    },
    {
      title: "Feature Selection",
      description: "Core variables tracked for analysis",
      details: [
        "Year and country identifiers",
        "Population statistics",
        "GDP indicators",
        "Primary energy consumption",
        "CO₂ emissions (oil, coal, total)",
        "Total greenhouse gas emissions",
        "Temperature change metrics"
      ]
    },
    {
      title: "Missing Value Treatment",
      description: "Systematic approach to handle gaps in data",
      details: [
        "Drop columns with >50% missing values (post-1950)",
        "Apply forward-fill within each country group",
        "Use backward-fill for remaining gaps",
        "Validate completeness of key indicators"
      ]
    },
    {
      title: "Quality Assurance",
      description: "Data validation and outlier review",
      details: [
        "Visual inspection of boxplots for key features",
        "Review of value distributions",
        "No outlier transformations applied",
        "Maintain original scale of measurements"
      ]
    }
  ];

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
            Our analysis utilizes the comprehensive Our World in Data CO₂ emissions dataset, 
            covering the period from 1950 to present.
          </p>
        </div>
      </motion.section>

      <section className="py-12">
        <div className="section-container">
          <Card className="mb-8">
            <CardHeader>
              <div className="flex items-center gap-2">
                <FileText className="h-5 w-5 text-green-600" />
                <CardTitle className="text-green-700">Data Source</CardTitle>
              </div>
              <CardDescription>
                Primary dataset: owid-co2-data.csv from Our World in Data
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h3 className="font-medium text-green-700">Key Features</h3>
                  <ul className="list-disc pl-5 space-y-2 text-gray-600">
                    <li>CO₂ emissions data (total, by source)</li>
                    <li>Greenhouse gas measurements</li>
                    <li>Economic indicators (GDP)</li>
                    <li>Energy consumption metrics</li>
                    <li>Population statistics</li>
                    <li>Temperature change data</li>
                  </ul>
                </div>
                <div className="space-y-4">
                  <h3 className="font-medium text-green-700">Coverage</h3>
                  <ul className="list-disc pl-5 space-y-2 text-gray-600">
                    <li>Time period: 1950 - Present</li>
                    <li>Country-level granularity</li>
                    <li>Annual measurements</li>
                    <li>Consistent ISO country codes</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="grid md:grid-cols-2 gap-8 mb-12">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
              viewport={{ once: true }}
            >
              <Card>
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <CircleCheck className="h-5 w-5 text-green-600" />
                    <CardTitle className="text-green-700">Data Quality Status</CardTitle>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-3 bg-green-50 rounded">
                      <span>Missing Value Treatment</span>
                      <span className="text-green-600">Complete</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-green-50 rounded">
                      <span>Time Range Filter</span>
                      <span className="text-green-600">Applied</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-green-50 rounded">
                      <span>Country Code Validation</span>
                      <span className="text-green-600">Verified</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-green-50 rounded">
                      <span>Outlier Review</span>
                      <span className="text-green-600">Completed</span>
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
                  <div className="flex items-center gap-2">
                    <AlertCircle className="h-5 w-5 text-amber-500" />
                    <CardTitle className="text-green-700">Data Insights</CardTitle>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="h-[250px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={cleanedData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="year" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line 
                          type="monotone" 
                          dataKey="co2" 
                          name="CO₂ Emissions"
                          stroke="#15803d" 
                          strokeWidth={2}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="energy"
                          name="Energy Consumption" 
                          stroke="#0284c7"
                          strokeWidth={2}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="mt-4 text-sm text-gray-600">
                    <Info className="inline-block mr-2 h-4 w-4" />
                    Trend visualization of key metrics after data cleaning
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </div>

          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Filter className="h-5 w-5 text-green-600" />
                  <CardTitle className="text-green-700">Data Preprocessing Steps</CardTitle>
                </div>
                <Button
                  onClick={() => setShowPreprocessing(!showPreprocessing)}
                  variant="outline"
                  className="text-green-700 border-green-600"
                >
                  {showPreprocessing ? 'Hide Details' : 'Show Details'}
                </Button>
              </div>
              <CardDescription>
                Systematic approach to prepare the dataset for analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              {showPreprocessing && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5 }}
                >
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
                </motion.div>
              )}

              <div className="mt-8">
                <h3 className="text-lg font-medium text-green-700 mb-4">Sample Cleaned Records</h3>
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Year</TableHead>
                        <TableHead>CO₂ Emissions</TableHead>
                        <TableHead>Energy Consumption</TableHead>
                        <TableHead>Population</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {cleanedData.map((row, index) => (
                        <TableRow key={index}>
                          <TableCell>{row.year}</TableCell>
                          <TableCell>{row.co2}</TableCell>
                          <TableCell>{row.energy}</TableCell>
                          <TableCell>{row.population}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>
    </div>
  );
};

export default Dataset;
