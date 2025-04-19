
import { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";
import { Leaf, BarChart3, Database, Layers, Play, Menu, X } from "lucide-react";
import { Button } from "@/components/ui/button";

const NavigationBar = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 10) {
        setIsScrolled(true);
      } else {
        setIsScrolled(false);
      }
    };

    window.addEventListener("scroll", handleScroll);
    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);

  const navLinks = [
    { name: "Home", path: "/", icon: <Leaf size={18} className="mr-1" /> },
    { name: "Features", path: "/features", icon: <Layers size={18} className="mr-1" /> },
    { name: "Dataset", path: "/dataset", icon: <Database size={18} className="mr-1" /> },
    { name: "Models", path: "/models", icon: <BarChart3 size={18} className="mr-1" /> },
    { name: "Demo", path: "/demo", icon: <Play size={18} className="mr-1" /> },
  ];

  const toggleNavbar = () => {
    setIsOpen(!isOpen);
  };

  return (
    <nav
      className={cn(
        "fixed top-0 left-0 w-full z-50 transition-all duration-300",
        isScrolled ? "bg-white/95 backdrop-blur-sm shadow-sm" : "bg-transparent"
      )}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center py-4 md:justify-start md:space-x-10">
          <div className="flex justify-start lg:w-0 lg:flex-1">
            <Link to="/" className="flex items-center">
              <Leaf className="h-8 w-8 text-green-600" />
              <span className="ml-2 text-xl font-bold text-green-700">Decarbonize-TH</span>
            </Link>
          </div>
          <div className="-mr-2 -my-2 md:hidden">
            <Button
              variant="ghost"
              onClick={toggleNavbar}
              className="rounded-md p-2 inline-flex items-center justify-center text-gray-600 hover:text-green-600 focus:outline-none"
            >
              {isOpen ? (
                <X className="h-6 w-6" aria-hidden="true" />
              ) : (
                <Menu className="h-6 w-6" aria-hidden="true" />
              )}
            </Button>
          </div>
          <div className="hidden md:flex items-center justify-end md:flex-1 lg:w-0">
            <div className="flex space-x-8">
              {navLinks.map((link) => (
                <Link
                  key={link.name}
                  to={link.path}
                  className={cn(
                    "inline-flex items-center px-1 pt-1 text-sm font-medium border-b-2 transition-colors duration-300",
                    location.pathname === link.path
                      ? "border-green-600 text-green-700"
                      : "border-transparent text-gray-600 hover:text-green-700 hover:border-green-300"
                  )}
                >
                  {link.icon}
                  {link.name}
                </Link>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Mobile menu, show/hide based on mobile menu state */}
      <div
        className={cn(
          "md:hidden",
          isOpen ? "block" : "hidden"
        )}
        id="mobile-menu"
      >
        <div className="pt-2 pb-3 space-y-1 bg-white/95 backdrop-blur-sm shadow-lg">
          {navLinks.map((link) => (
            <Link
              key={link.name}
              to={link.path}
              className={cn(
                "flex items-center px-3 py-2 text-base font-medium border-l-4",
                location.pathname === link.path
                  ? "bg-green-50 border-green-500 text-green-700"
                  : "border-transparent text-gray-600 hover:bg-green-50 hover:border-green-300 hover:text-green-700"
              )}
              onClick={() => setIsOpen(false)}
            >
              <span className="mr-2">{link.icon}</span>
              {link.name}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
};

export default NavigationBar;
