
import React, { useRef } from 'react';
import { motion, useScroll, useTransform } from "framer-motion";
import { Filter, Database, ChartBar, Users, Thermometer } from "lucide-react";
import FeatureTable from '@/components/features/FeatureTable';
import CorrelationHeatmap from '@/components/features/CorrelationHeatmap';
import FeatureInsights from '@/components/features/FeatureInsights';
import FeatureCorrelationChart from '@/components/features/FeatureCorrelationChart';

const Features = () => {
  const scrollRef = useRef(null);
  const { scrollYProgress } = useScroll({
    target: scrollRef,
    offset: ["start start", "end start"]
  });
  
  const opacity = useTransform(scrollYProgress, [0, 0.5], [1, 0]);
  const scale = useTransform(scrollYProgress, [0, 0.5], [1, 0.95]);

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
              <Filter className="h-8 w-8 text-green-600" />
            </div>
            <h1 className="text-4xl font-bold text-green-800 mb-6">Feature Analysis</h1>
            <p className="text-lg text-gray-700 max-w-3xl mx-auto mb-8">
              Discover the key predictors of CO₂ emissions in Thailand, refined through 
              statistical analysis and feature selection techniques.
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
          <div className="grid grid-cols-1 gap-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              viewport={{ once: true }}
            >
              <div className="flex items-center mb-6">
                <div className="bg-green-100 p-2 rounded-full mr-3">
                  <Database className="h-5 w-5 text-green-600" />
                </div>
                <h2 className="text-2xl font-bold text-green-800">Final Prediction Features</h2>
              </div>
              <FeatureTable className="shadow-md" />
            </motion.div>
            
            <div className="grid md:grid-cols-2 gap-8">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1 }}
                viewport={{ once: true }}
              >
                <div className="flex items-center mb-6">
                  <div className="bg-green-100 p-2 rounded-full mr-3">
                    <ChartBar className="h-5 w-5 text-green-600" />
                  </div>
                  <h2 className="text-2xl font-bold text-green-800">Feature Impact</h2>
                </div>
                <FeatureCorrelationChart className="shadow-md h-full" />
              </motion.div>
              
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
                viewport={{ once: true }}
              >
                <div className="flex items-center mb-6">
                  <div className="bg-green-100 p-2 rounded-full mr-3">
                    <Thermometer className="h-5 w-5 text-green-600" />
                  </div>
                  <h2 className="text-2xl font-bold text-green-800">Correlation Insights</h2>
                </div>
                <FeatureInsights className="shadow-md h-full" />
              </motion.div>
            </div>
          </div>
        </div>
      </section>
      
      <section className="py-12 bg-green-50">
        <div className="section-container">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            viewport={{ once: true }}
          >
            <div className="flex items-center mb-6">
              <div className="bg-green-100 p-2 rounded-full mr-3">
                <Users className="h-5 w-5 text-green-600" />
              </div>
              <h2 className="text-2xl font-bold text-green-800">Feature Correlation Matrix</h2>
            </div>
            <p className="text-gray-700 mb-6 max-w-3xl">
              This heatmap visualizes how different features correlate with each other and with CO₂ emissions.
              Stronger colors indicate stronger relationships, with green showing positive correlation and 
              red showing negative correlation.
            </p>
            <CorrelationHeatmap className="shadow-md" />
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default Features;