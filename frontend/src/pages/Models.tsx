
import { motion } from "framer-motion";
import { BarChart3 } from "lucide-react";
import ModelTabs from "@/components/models/ModelTabs";
import { modelData, getBestModel } from "@/utils/modelData";

const Models = () => {
  const bestModel = getBestModel(modelData);

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
            <BarChart3 className="h-8 w-8 text-green-600" />
          </div>
          <h1 className="text-4xl font-bold text-green-800 mb-6">Model Performance Analysis</h1>
          <p className="text-lg text-gray-700 max-w-3xl mx-auto mb-8">
            Compare and analyze the performance of different machine learning models 
            for predicting CO₂ emissions in Thailand.
          </p>
        </div>
      </motion.section>

      <section className="py-12">
        <div className="section-container">
          <ModelTabs models={modelData} bestModelId={bestModel.id} />
        </div>
      </section>
    </div>
  );
};

export default Models;
