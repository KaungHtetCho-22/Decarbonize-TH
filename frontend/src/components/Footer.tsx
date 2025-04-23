
import { Link } from "react-router-dom";
import { Leaf, Github, ExternalLink } from "lucide-react";

const Footer = () => {
  return (
    <footer className="bg-gray-50 border-t border-gray-200 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <Link to="/" className="flex items-center">
              <Leaf className="h-6 w-6 text-green-600" />
              <span className="ml-2 text-xl font-bold text-green-700">Decarbonize-TH</span>
            </Link>
            <p className="mt-4 text-sm text-gray-600">
              A machine learning project focused on predicting carbon emissions in Thailand to support sustainability goals.
            </p>
          </div>
          
          <div>
            <h3 className="text-sm font-semibold text-gray-700 tracking-wider uppercase">Navigation</h3>
            <ul className="mt-4 space-y-3">
              {[
                { name: "Home", path: "/" },
                { name: "Features", path: "/features" },
                { name: "Dataset", path: "/dataset" },
                { name: "Models", path: "/models" },
                { name: "Demo", path: "/demo" },
              ].map((item) => (
                <li key={item.name}>
                  <Link 
                    to={item.path} 
                    className="text-base text-gray-600 hover:text-green-700"
                  >
                    {item.name}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
          
          <div>
            <h3 className="text-sm font-semibold text-gray-700 tracking-wider uppercase">Project</h3>
            <ul className="mt-4 space-y-3">
              <li>
                <a
                  href="https://github.com/KaungHtetCho-22/Decarbonize-TH"
                  className="text-base text-gray-600 hover:text-green-700 flex items-center"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Github className="h-4 w-4 mr-2" />
                  GitHub Repository
                  <ExternalLink className="h-3 w-3 ml-1" />
                </a>
              </li>
              <li>
                {/* <a
                  href="#"
                  className="text-base text-gray-600 hover:text-green-700 flex items-center"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <ExternalLink className="h-4 w-4 mr-2" />
                  Related Research
                </a> */}
              </li>
            </ul>
          </div>
        </div>
        <div className="mt-12 border-t border-gray-200 pt-8">
          <p className="text-sm text-center text-gray-500">
            &copy; {new Date().getFullYear()} Decarbonize-TH Project. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
