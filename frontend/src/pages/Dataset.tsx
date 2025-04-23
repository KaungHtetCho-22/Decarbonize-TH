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
import { Database, Filter, CircleCheck, Info, FileText, LineChart, BarChart3, PieChart } from "lucide-react";

// --- New chart cards for maintainability
import ThailandTrendDashboardCard from "@/components/dataset/ThailandTrendDashboardCard";
import YearlyMissingRatioCard from "@/components/dataset/YearlyMissingRatioCard";

// --- USE: latest requested "sample cleaned records"
const cleanedData = [
  {
    year: 1950, population: 20428403.0, gdp: 26094198784.0, cement_co2_per_capita: 0.004, co2_growth_abs: 0.894,
    co2_including_luc_growth_abs: -2.338, co2_including_luc_per_gdp: 8.579, co2_including_luc_per_unit_energy: 8.235,
    co2_per_gdp: 0.037, co2_per_unit_energy: 0.242, coal_co2_per_capita: 0.002, energy_per_capita: 992.445,
    flaring_co2_per_capita: 0.003, nitrous_oxide_per_capita: 0.288, temperature_change_from_n2o: 0.0,
    country: "Thailand", co2: 0.956
  },
  {
    year: 1951, population: 20965189.0, gdp: 27943960576.0, cement_co2_per_capita: 0.007, co2_growth_abs: 0.135,
    co2_including_luc_growth_abs: 0.458, co2_including_luc_per_gdp: 8.028, co2_including_luc_per_unit_energy: 8.235,
    co2_per_gdp: 0.039, co2_per_unit_energy: 0.242, coal_co2_per_capita: 0.002, energy_per_capita: 992.445,
    flaring_co2_per_capita: 0.003, nitrous_oxide_per_capita: 0.282, temperature_change_from_n2o: 0.0,
    country: "Thailand", co2: 1.091
  },
  {
    year: 1952, population: 21527581.0, gdp: 29485821952.0, cement_co2_per_capita: 0.006, co2_growth_abs: 0.187,
    co2_including_luc_growth_abs: 7.96, co2_including_luc_per_gdp: 7.878, co2_including_luc_per_unit_energy: 8.235,
    co2_per_gdp: 0.043, co2_per_unit_energy: 0.242, coal_co2_per_capita: 0.003, energy_per_capita: 992.445,
    flaring_co2_per_capita: 0.003, nitrous_oxide_per_capita: 0.277, temperature_change_from_n2o: 0.0,
    country: "Thailand", co2: 1.278
  },
  {
    year: 1953, population: 22109322.0, gdp: 32726595584.0, cement_co2_per_capita: 0.006, co2_growth_abs: 0.366,
    co2_including_luc_growth_abs: 4.407, co2_including_luc_per_gdp: 7.233, co2_including_luc_per_unit_energy: 8.235,
    co2_per_gdp: 0.05, co2_per_unit_energy: 0.242, coal_co2_per_capita: 0.004, energy_per_capita: 992.445,
    flaring_co2_per_capita: 0.003, nitrous_oxide_per_capita: 0.272, temperature_change_from_n2o: 0.0,
    country: "Thailand", co2: 1.644
  },
  {
    year: 1954, population: 22713342.0, gdp: 32462198784.0, cement_co2_per_capita: 0.008, co2_growth_abs: 0.406,
    co2_including_luc_growth_abs: 6.434, co2_including_luc_per_gdp: 7.49, co2_including_luc_per_unit_energy: 8.235,
    co2_per_gdp: 0.063, co2_per_unit_energy: 0.242, coal_co2_per_capita: 0.009, energy_per_capita: 992.445,
    flaring_co2_per_capita: 0.003, nitrous_oxide_per_capita: 0.268, temperature_change_from_n2o: 0.0,
    country: "Thailand", co2: 2.05
  },
];

// KEEP ONLY THE RELEVANT STEPS (REMOVE FEATURE SELECTION STEP)
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
  // REMOVE THE "Feature Selection" step here
  {
    title: "Missing Value Treatment",
    description: "Systematic approach to handle gaps in data",
    details: [
      "Drop columns with >50% missing values (post-1950)",
      "Apply forward-fill within each country group",
      "Use backward-fill for remaining gaps",
      "Validate completeness of key indicators"
    ]
  }
];

// Animation config
const fadeIn = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6 }
};

const Dataset = () => {
  const [showPreprocessing, setShowPreprocessing] = useState(false);

  return (
    <div className="min-h-screen bg-green-50/30">
      <motion.section 
        className="relative py-12 bg-gradient-to-b from-green-50 to-white overflow-hidden"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6 }}
      >
        <div className="section-container text-center">
          <div className="inline-block p-2 bg-green-100 rounded-full mb-4">
            <Database className="h-8 w-8 text-green-600" />
          </div>
          <h1 className="text-4xl font-bold text-green-800 mb-6">Dataset Overview</h1>
          <p className="text-lg text-gray-700 max-w-3xl mx-auto mb-4">
            Our analysis utilizes the comprehensive Our World in Data CO₂ emissions dataset, 
            covering the period from 1750 to 2023 (used only from 1950 onward due to missingness).
          </p>
          <a
            href="https://ourworldindata.org/co2-emissions"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-block text-green-700 hover:text-green-800 underline mb-4"
          >
            <span className="font-semibold underline">View original dataset on Our World in Data</span>
          </a>
        </div>
      </motion.section>

      <section className="py-8 bg-green-50/30">
        <div className="section-container">

          {/* Dashboard + Missing Ratio on grid w/ responsive adjustment */}
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2 mb-8">
            <motion.div {...fadeIn} className="col-span-1">
              <ThailandTrendDashboardCard />
            </motion.div>
            <motion.div {...fadeIn} className="col-span-1">
              <YearlyMissingRatioCard />
            </motion.div>
          </div>

          {/* Dataset Split Summary */}
          <motion.div {...fadeIn} className="mb-8">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-xl text-green-700">Train-Validation-Test Split</CardTitle>
                <CardDescription>
                  Row Counts by Dataset Split
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Split</TableHead>
                        <TableHead className="text-right">Rows</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      <TableRow>
                        <TableCell>Global Train (World excl. ASEAN + Thailand)</TableCell>
                        <TableCell className="text-right">15,392</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>ASEAN Train (60%)</TableCell>
                        <TableCell className="text-right">399</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>ASEAN Val (40%)</TableCell>
                        <TableCell className="text-right">267</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Thailand Test</TableCell>
                        <TableCell className="text-right">74</TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* REMOVE Status + Overview cards grid! (Data Quality Status, Feature Selection) */}
          {/* Removed grid/md:grid-cols-2/Status & Feature cards */}

          {/* Data Preprocessing Steps */}
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
                        <TableHead>Population</TableHead>
                        <TableHead>GDP</TableHead>
                        <TableHead>Cement CO₂/Capita</TableHead>
                        <TableHead>CO₂ Growth Abs</TableHead>
                        <TableHead>CO₂+LUC Growth Abs</TableHead>
                        <TableHead>CO₂+LUC/GDP</TableHead>
                        <TableHead>CO₂+LUC/Energy</TableHead>
                        <TableHead>CO₂/GDP</TableHead>
                        <TableHead>CO₂/Energy</TableHead>
                        <TableHead>Coal CO₂/Capita</TableHead>
                        <TableHead>Energy/Capita</TableHead>
                        <TableHead>Flaring CO₂/Capita</TableHead>
                        <TableHead>N₂O/Capita</TableHead>
                        <TableHead>Temp Δ from N₂O</TableHead>
                        <TableHead>CO₂</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {cleanedData.map((row, index) => (
                        <TableRow key={index}>
                          <TableCell>{row.year}</TableCell>
                          <TableCell>{row.population.toLocaleString()}</TableCell>
                          <TableCell>{row.gdp.toExponential(3)}</TableCell>
                          <TableCell>{row.cement_co2_per_capita}</TableCell>
                          <TableCell>{row.co2_growth_abs}</TableCell>
                          <TableCell>{row.co2_including_luc_growth_abs}</TableCell>
                          <TableCell>{row.co2_including_luc_per_gdp}</TableCell>
                          <TableCell>{row.co2_including_luc_per_unit_energy}</TableCell>
                          <TableCell>{row.co2_per_gdp}</TableCell>
                          <TableCell>{row.co2_per_unit_energy}</TableCell>
                          <TableCell>{row.coal_co2_per_capita}</TableCell>
                          <TableCell>{row.energy_per_capita}</TableCell>
                          <TableCell>{row.flaring_co2_per_capita}</TableCell>
                          <TableCell>{row.nitrous_oxide_per_capita}</TableCell>
                          <TableCell>{row.temperature_change_from_n2o}</TableCell>
                          <TableCell>{row.co2}</TableCell>
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
