
import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { ChevronRight, Leaf, BarChart3, Database, Layers } from "lucide-react";

const Home = () => {
  const [scrollY, setScrollY] = useState(0);

  useEffect(() => {
    const handleScroll = () => {
      setScrollY(window.scrollY);
    };
    
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <div className="min-h-screen">
      {/* Hero section */}
      <section className="pt-24 pb-12 md:pt-28 md:pb-20 relative overflow-hidden">
        <div 
          className="absolute inset-0 z-0 opacity-30"
          style={{
            background: "linear-gradient(90deg, hsla(139, 70%, 75%, 1) 0%, hsla(63, 90%, 76%, 1) 100%)",
            backgroundSize: "200px 200px",
            filter: "blur(100px)"
          }}
        />
        
        <div className="section-container relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center"
          >
            <motion.div
              className="inline-block p-2 bg-green-100 rounded-full mb-4"
              animate={{ y: [0, -5, 0] }}
              transition={{ repeat: Infinity, duration: 2 }}
            >
              <Leaf className="h-8 w-8 text-green-600" />
            </motion.div>
            
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-green-800 mb-6">
              Decarbonize-
              <span className="text-green-600">TH</span>
            </h1>
            
            <p className="text-lg md:text-xl text-gray-700 max-w-3xl mx-auto mb-8">
              A machine learning project predicting carbon emissions in Thailand, 
              helping to build a more sustainable future through data-driven insights.
            </p>
            
            <div className="flex flex-col sm:flex-row justify-center items-center space-y-4 sm:space-y-0 sm:space-x-4">
              <Button 
                size="lg"
                className="bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-6 rounded-md shadow-md hover:shadow-lg transition-all"
                asChild
              >
                <Link to="/demo">
                  Try the Demo <ChevronRight className="ml-2 h-5 w-5" />
                </Link>
              </Button>
              
              <Button 
                variant="outline" 
                size="lg"
                className="border-green-600 text-green-700 font-semibold hover:bg-green-50"
                asChild
              >
                <Link to="/features">
                  Learn More
                </Link>
              </Button>
            </div>
          </motion.div>
        </div>
        
        {/* Animated leaves background */}
        <div className="absolute inset-0 z-0 overflow-hidden">
          {[...Array(10)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute text-green-500 opacity-20"
              initial={{
                x: Math.random() * 100,
                y: -20,
                rotate: 0,
              }}
              animate={{
                x: Math.random() * 100 + i * 100,
                y: window.innerHeight,
                rotate: 360,
              }}
              transition={{
                duration: 15 + Math.random() * 10,
                repeat: Infinity,
                ease: "linear",
                delay: i * 2,
              }}
            >
              <Leaf size={20 + Math.random() * 20} />
            </motion.div>
          ))}
        </div>
      </section>
      
      {/* About Project Section */}
      <section className="py-12 md:py-20 bg-green-50">
        <div className="section-container">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="section-title">About the Project</h2>
            <div className="grid md:grid-cols-2 gap-8 mt-8">
              <div>
                <p className="text-gray-600 mb-4">
                  Decarbonize-TH is a machine learning initiative designed to predict and analyze carbon emissions 
                  across Thailand. By combining historical data with advanced predictive models, we aim to provide 
                  valuable insights for policy makers, businesses, and individuals committed to reducing their 
                  carbon footprint.
                </p>
                <p className="text-gray-600 mb-4">
                  Our project leverages sophisticated algorithms to identify patterns and trends in emission data, 
                  allowing for more accurate forecasting and targeted reduction strategies.
                </p>
              </div>
              <div className="bg-white p-6 rounded-lg shadow-md">
                <h3 className="text-xl font-semibold text-green-700 mb-4">Key Objectives</h3>
                <ul className="space-y-3">
                  {[
                    "Predict carbon emissions across different sectors in Thailand",
                    "Identify key factors contributing to emission levels",
                    "Provide data-driven recommendations for emission reduction",
                    "Support Thailand's commitment to sustainability goals",
                    "Create accessible tools for businesses and policy makers"
                  ].map((item, index) => (
                    <motion.li 
                      key={index}
                      className="flex items-start"
                      initial={{ opacity: 0, x: -20 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      viewport={{ once: true }}
                    >
                      <span className="inline-flex items-center justify-center h-6 w-6 rounded-full bg-green-100 text-green-600 mr-3 flex-shrink-0">
                        <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      </span>
                      <span className="text-gray-700">{item}</span>
                    </motion.li>
                  ))}
                </ul>
              </div>
            </div>
          </motion.div>
        </div>
      </section>
      
      {/* Project Overview Cards */}
      <section className="py-12 md:py-20">
        <div className="section-container">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="section-title">Project Components</h2>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mt-8">
              {[
                {
                  title: "Data Analysis",
                  icon: <Database className="h-10 w-10 text-green-600 mb-4" />,
                  description: "Comprehensive analysis of carbon emission factors across different sectors in Thailand",
                  link: "/dataset"
                },
                {
                  title: "Feature Engineering",
                  icon: <Layers className="h-10 w-10 text-green-600 mb-4" />,
                  description: "Advanced feature extraction and transformation to create predictive variables",
                  link: "/features"
                },
                {
                  title: "Predictive Models",
                  icon: <BarChart3 className="h-10 w-10 text-green-600 mb-4" />,
                  description: "Machine learning models trained to forecast carbon emissions with high accuracy",
                  link: "/models"
                }
              ].map((card, index) => (
                <motion.div
                  key={index}
                  className="card-green p-6"
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  viewport={{ once: true }}
                  whileHover={{ y: -5, transition: { duration: 0.2 } }}
                >
                  {card.icon}
                  <h3 className="text-xl font-semibold text-green-700 mb-2">{card.title}</h3>
                  <p className="text-gray-600 mb-4">{card.description}</p>
                  <Link 
                    to={card.link} 
                    className="text-green-600 font-medium hover:text-green-700 inline-flex items-center"
                  >
                    Learn more
                    <ChevronRight className="ml-1 h-4 w-4" />
                  </Link>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>
      
      {/* Call to Action */}
      <section className="py-12 md:py-16 bg-green-600 text-white">
        <div className="section-container text-center">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-3xl md:text-4xl font-bold mb-6">Ready to explore carbon emission predictions?</h2>
            <p className="text-lg md:text-xl text-green-100 max-w-2xl mx-auto mb-8">
              Try our interactive demo to see how different factors affect carbon emissions in Thailand.
            </p>
            <Button 
              size="lg"
              className="bg-white text-green-700 hover:bg-green-50 font-semibold py-3 px-8 rounded-md shadow-md"
              asChild
            >
              <Link to="/demo">
                Try the Demo <ChevronRight className="ml-2 h-5 w-5" />
              </Link>
            </Button>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default Home;
