import { motion } from "framer-motion";
import { BarChart3 } from "lucide-react";

const HeroSection = () => (
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
      <h1 className="text-4xl font-bold text-green-800 mb-6">Predictive Models</h1>
      <p className="text-lg text-gray-700 max-w-3xl mx-auto mb-8">
        Explore the machine learning models we've developed to predict carbon emissions in Thailand, 
        and learn how we selected the best performing approach.
      </p>
    </div>
    
    <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
      {[...Array(20)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute bg-green-600 opacity-10"
          style={{
            height: 2 + Math.random() * 40,
            width: 2,
            top: Math.random() * 100 + '%',
            left: Math.random() * 100 + '%',
          }}
          animate={{
            height: [2 + Math.random() * 40, 60 + Math.random() * 80, 2 + Math.random() * 40],
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
);

export default HeroSection;
