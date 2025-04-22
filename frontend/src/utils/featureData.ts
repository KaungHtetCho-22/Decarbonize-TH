
export interface FeatureInfo {
  name: string;
  vif: number;
  correlation: number;
  description: string;
}

export const featureData: FeatureInfo[] = [
  {
    name: "gdp",
    vif: 11.51,
    correlation: 0.93,
    description: "Core driver of emissions via economic activity."
  },
  {
    name: "temperature_change_from_n2o",
    vif: 12.03,
    correlation: 0.86,
    description: "Captures global warming patterns tied to GHGs."
  },
  {
    name: "population",
    vif: 3.94,
    correlation: 0.66,
    description: "More people means more emissions."
  },
  {
    name: "co2_growth_abs",
    vif: 2.09,
    correlation: 0.46,
    description: "Captures rapid changes in emission trend."
  },
  {
    name: "coal_co2_per_capita",
    vif: 1.66,
    correlation: 0.23,
    description: "Fossil-specific intensity contributor."
  },
  {
    name: "co2_including_luc_growth_abs",
    vif: 1.45,
    correlation: 0.23,
    description: "Adds land-use change to emission dynamics."
  },
  {
    name: "cement_co2_per_capita",
    vif: 1.61,
    correlation: 0.18,
    description: "Industrial emissions proxy for construction."
  },
  {
    name: "energy_per_capita",
    vif: 2.81,
    correlation: 0.17,
    description: "Captures consumption behavior."
  },
  {
    name: "co2_per_gdp",
    vif: 2.37,
    correlation: 0.05,
    description: "Emission efficiency indicator."
  },
  {
    name: "nitrous_oxide_per_capita",
    vif: 2.16,
    correlation: 0.01,
    description: "GHG context, moderate in emission impact."
  },
  {
    name: "co2_per_unit_energy",
    vif: 1.63,
    correlation: 0.00,
    description: "Energy system carbon intensity."
  },
  {
    name: "flaring_co2_per_capita",
    vif: 1.78,
    correlation: -0.01,
    description: "Industrial inefficiency metric."
  },
  {
    name: "co2_including_luc_per_unit_energy",
    vif: 2.22,
    correlation: -0.02,
    description: "Emissions per unit energy including land-use."
  },
  {
    name: "co2_including_luc_per_gdp",
    vif: 1.84,
    correlation: -0.06,
    description: "Overall carbon efficiency when land is factored."
  }
];

export const formatFeatureName = (name: string): string => {
  return name
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

export interface CorrelationPoint {
  x: string;
  y: string;
  value: number;
}

export const correlationData: CorrelationPoint[] = [
  { "x": "population", "y": "population", "value": 1.0 },
  { "x": "population", "y": "gdp", "value": 0.64 },
  { "x": "population", "y": "cement_co2_per_capita", "value": 0.07 },
  { "x": "population", "y": "co2_growth_abs", "value": 0.48 },
  { "x": "population", "y": "co2_including_luc_growth_abs", "value": 0.24 },
  { "x": "population", "y": "co2_including_luc_per_gdp", "value": -0.06 },
  { "x": "population", "y": "co2_including_luc_per_unit_energy", "value": -0.02 },
  { "x": "population", "y": "co2_per_gdp", "value": 0.01 },
  { "x": "population", "y": "co2_per_unit_energy", "value": 0.0 },
  { "x": "population", "y": "coal_co2_per_capita", "value": 0.07 },
  { "x": "population", "y": "energy_per_capita", "value": -0.02 },
  { "x": "population", "y": "flaring_co2_per_capita", "value": -0.02 },
  { "x": "population", "y": "nitrous_oxide_per_capita", "value": -0.05 },
  { "x": "population", "y": "temperature_change_from_n2o", "value": 0.68 },
  { "x": "population", "y": "co2", "value": 0.66 },
  { "x": "gdp", "y": "population", "value": 0.64 },
  { "x": "gdp", "y": "gdp", "value": 1.0 },
  { "x": "gdp", "y": "cement_co2_per_capita", "value": 0.17 },
  { "x": "gdp", "y": "co2_growth_abs", "value": 0.34 },
  { "x": "gdp", "y": "co2_including_luc_growth_abs", "value": 0.17 },
  { "x": "gdp", "y": "co2_including_luc_per_gdp", "value": -0.07 },
  { "x": "gdp", "y": "co2_including_luc_per_unit_energy", "value": -0.02 },
  { "x": "gdp", "y": "co2_per_gdp", "value": 0.01 },
  { "x": "gdp", "y": "co2_per_unit_energy", "value": -0.02 },
  { "x": "gdp", "y": "coal_co2_per_capita", "value": 0.18 },
  { "x": "gdp", "y": "energy_per_capita", "value": 0.15 },
  { "x": "gdp", "y": "flaring_co2_per_capita", "value": -0.02 },
  { "x": "gdp", "y": "nitrous_oxide_per_capita", "value": -0.0 },
  { "x": "gdp", "y": "temperature_change_from_n2o", "value": 0.89 },
  { "x": "gdp", "y": "co2", "value": 0.93 },
  { "x": "cement_co2_per_capita", "y": "population", "value": 0.07 },
  { "x": "cement_co2_per_capita", "y": "gdp", "value": 0.17 },
  { "x": "cement_co2_per_capita", "y": "cement_co2_per_capita", "value": 1.0 },
  { "x": "cement_co2_per_capita", "y": "co2_growth_abs", "value": 0.09 },
  { "x": "cement_co2_per_capita", "y": "co2_including_luc_growth_abs", "value": 0.04 },
  { "x": "cement_co2_per_capita", "y": "co2_including_luc_per_gdp", "value": -0.16 },
  { "x": "cement_co2_per_capita", "y": "co2_including_luc_per_unit_energy", "value": -0.07 },
  { "x": "cement_co2_per_capita", "y": "co2_per_gdp", "value": 0.13 },
  { "x": "cement_co2_per_capita", "y": "co2_per_unit_energy", "value": -0.04 },
  { "x": "cement_co2_per_capita", "y": "coal_co2_per_capita", "value": 0.28 },
  { "x": "cement_co2_per_capita", "y": "energy_per_capita", "value": 0.47 },
  { "x": "cement_co2_per_capita", "y": "flaring_co2_per_capita", "value": 0.01 },
  { "x": "cement_co2_per_capita", "y": "nitrous_oxide_per_capita", "value": 0.05 },
  { "x": "cement_co2_per_capita", "y": "temperature_change_from_n2o", "value": 0.11 },
  { "x": "cement_co2_per_capita", "y": "co2", "value": 0.18 },
  { "x": "co2_growth_abs", "y": "population", "value": 0.48 },
  { "x": "co2_growth_abs", "y": "gdp", "value": 0.34 },
  { "x": "co2_growth_abs", "y": "cement_co2_per_capita", "value": 0.09 },
  { "x": "co2_growth_abs", "y": "co2_growth_abs", "value": 1.0 },
  { "x": "co2_growth_abs", "y": "co2_including_luc_growth_abs", "value": 0.54 },
  { "x": "co2_growth_abs", "y": "co2_including_luc_per_gdp", "value": 0.0 },
  { "x": "co2_growth_abs", "y": "co2_including_luc_per_unit_energy", "value": -0.01 },
  { "x": "co2_growth_abs", "y": "co2_per_gdp", "value": 0.14 },
  { "x": "co2_growth_abs", "y": "co2_per_unit_energy", "value": 0.06 },
  { "x": "co2_growth_abs", "y": "coal_co2_per_capita", "value": 0.07 },
  { "x": "co2_growth_abs", "y": "energy_per_capita", "value": 0.03 },
  { "x": "co2_growth_abs", "y": "flaring_co2_per_capita", "value": 0.0 },
  { "x": "co2_growth_abs", "y": "nitrous_oxide_per_capita", "value": -0.01 },
  { "x": "co2_growth_abs", "y": "temperature_change_from_n2o", "value": 0.31 },
  { "x": "co2_growth_abs", "y": "co2", "value": 0.46 },
  { "x": "co2_including_luc_growth_abs", "y": "population", "value": 0.24 },
  { "x": "co2_including_luc_growth_abs", "y": "gdp", "value": 0.17 },
  { "x": "co2_including_luc_growth_abs", "y": "cement_co2_per_capita", "value": 0.04 },
  { "x": "co2_including_luc_growth_abs", "y": "co2_growth_abs", "value": 0.54 },
  { "x": "co2_including_luc_growth_abs", "y": "co2_including_luc_growth_abs", "value": 1.0 },
  { "x": "co2_including_luc_growth_abs", "y": "co2_including_luc_per_gdp", "value": 0.0 },
  { "x": "co2_including_luc_growth_abs", "y": "co2_including_luc_per_unit_energy", "value": -0.0 },
  { "x": "co2_including_luc_growth_abs", "y": "co2_per_gdp", "value": 0.08 },
  { "x": "co2_including_luc_growth_abs", "y": "co2_per_unit_energy", "value": 0.04 },
  { "x": "co2_including_luc_growth_abs", "y": "coal_co2_per_capita", "value": 0.04 },
  { "x": "co2_including_luc_growth_abs", "y": "energy_per_capita", "value": 0.01 },
  { "x": "co2_including_luc_growth_abs", "y": "flaring_co2_per_capita", "value": 0.0 },
  { "x": "co2_including_luc_growth_abs", "y": "nitrous_oxide_per_capita", "value": -0.0 },
  { "x": "co2_including_luc_growth_abs", "y": "temperature_change_from_n2o", "value": 0.15 },
  { "x": "co2_including_luc_growth_abs", "y": "co2", "value": 0.23 },
  { "x": "co2_including_luc_per_gdp", "y": "population", "value": -0.06 },
  { "x": "co2_including_luc_per_gdp", "y": "gdp", "value": -0.07 },
  { "x": "co2_including_luc_per_gdp", "y": "cement_co2_per_capita", "value": -0.16 },
  { "x": "co2_including_luc_per_gdp", "y": "co2_growth_abs", "value": 0.0 },
  { "x": "co2_including_luc_per_gdp", "y": "co2_including_luc_growth_abs", "value": 0.0 },
  { "x": "co2_including_luc_per_gdp", "y": "co2_including_luc_per_gdp", "value": 1.0 },
  { "x": "co2_including_luc_per_gdp", "y": "co2_including_luc_per_unit_energy", "value": 0.05 },
  { "x": "co2_including_luc_per_gdp", "y": "co2_per_gdp", "value": 0.11 },
  { "x": "co2_including_luc_per_gdp", "y": "co2_per_unit_energy", "value": 0.04 },
  { "x": "co2_including_luc_per_gdp", "y": "coal_co2_per_capita", "value": -0.09 },
  { "x": "co2_including_luc_per_gdp", "y": "energy_per_capita", "value": -0.12 },
  { "x": "co2_including_luc_per_gdp", "y": "flaring_co2_per_capita", "value": -0.01 },
  { "x": "co2_including_luc_per_gdp", "y": "nitrous_oxide_per_capita", "value": 0.22 },
  { "x": "co2_including_luc_per_gdp", "y": "temperature_change_from_n2o", "value": -0.07 },
  { "x": "co2_including_luc_per_gdp", "y": "co2", "value": -0.06 },
  { "x": "co2_including_luc_per_unit_energy", "y": "population", "value": -0.02 },
  { "x": "co2_including_luc_per_unit_energy", "y": "gdp", "value": -0.02 },
  { "x": "co2_including_luc_per_unit_energy", "y": "cement_co2_per_capita", "value": -0.07 },
  { "x": "co2_including_luc_per_unit_energy", "y": "co2_growth_abs", "value": -0.01 },
  { "x": "co2_including_luc_per_unit_energy", "y": "co2_including_luc_growth_abs", "value": -0.0 },
  { "x": "co2_including_luc_per_unit_energy", "y": "co2_including_luc_per_gdp", "value": 0.05 },
  { "x": "co2_including_luc_per_unit_energy", "y": "co2_including_luc_per_unit_energy", "value": 1.0 },
  { "x": "co2_including_luc_per_unit_energy", "y": "co2_per_gdp", "value": -0.04 },
  { "x": "co2_including_luc_per_unit_energy", "y": "co2_per_unit_energy", "value": 0.03 },
  { "x": "co2_including_luc_per_unit_energy", "y": "coal_co2_per_capita", "value": -0.05 },
  { "x": "co2_including_luc_per_unit_energy", "y": "energy_per_capita", "value": 0.0 },
  { "x": "co2_including_luc_per_unit_energy", "y": "flaring_co2_per_capita", "value": -0.01 },
  { "x": "co2_including_luc_per_unit_energy", "y": "nitrous_oxide_per_capita", "value": -0.03 },
  { "x": "co2_including_luc_per_unit_energy", "y": "temperature_change_from_n2o", "value": -0.02 },
  { "x": "co2_including_luc_per_unit_energy", "y": "co2", "value": -0.02 },
  { "x": "co2_per_gdp", "y": "population", "value": 0.01 },
  { "x": "co2_per_gdp", "y": "gdp", "value": 0.01 },
  { "x": "co2_per_gdp", "y": "cement_co2_per_capita", "value": 0.13 },
  { "x": "co2_per_gdp", "y": "co2_growth_abs", "value": 0.14 },
  { "x": "co2_per_gdp", "y": "co2_including_luc_growth_abs", "value": 0.08 },
  { "x": "co2_per_gdp", "y": "co2_including_luc_per_gdp", "value": 0.11 },
  { "x": "co2_per_gdp", "y": "co2_including_luc_per_unit_energy", "value": -0.04 },
  { "x": "co2_per_gdp", "y": "co2_per_gdp", "value": 1.0 },
  { "x": "co2_per_gdp", "y": "co2_per_unit_energy", "value": 0.31 },
  { "x": "co2_per_gdp", "y": "coal_co2_per_capita", "value": 0.22 },
  { "x": "co2_per_gdp", "y": "energy_per_capita", "value": 0.21 },
  { "x": "co2_per_gdp", "y": "flaring_co2_per_capita", "value": 0.07 },
  { "x": "co2_per_gdp", "y": "nitrous_oxide_per_capita", "value": 0.06 },
  { "x": "co2_per_gdp", "y": "temperature_change_from_n2o", "value": 0.02 },
  { "x": "co2_per_gdp", "y": "co2", "value": 0.05 },
  { "x": "co2_per_unit_energy", "y": "population", "value": 0.0 },
  { "x": "co2_per_unit_energy", "y": "gdp", "value": -0.02 },
  { "x": "co2_per_unit_energy", "y": "cement_co2_per_capita", "value": -0.04 },
  { "x": "co2_per_unit_energy", "y": "co2_growth_abs", "value": 0.06 },
  { "x": "co2_per_unit_energy", "y": "co2_including_luc_growth_abs", "value": 0.04 },
  { "x": "co2_per_unit_energy", "y": "co2_including_luc_per_gdp", "value": 0.04 },
  { "x": "co2_per_unit_energy", "y": "co2_including_luc_per_unit_energy", "value": 0.03 },
  { "x": "co2_per_unit_energy", "y": "co2_per_gdp", "value": 0.31 },
  { "x": "co2_per_unit_energy", "y": "co2_per_unit_energy", "value": 1.0 },
  { "x": "co2_per_unit_energy", "y": "coal_co2_per_capita", "value": 0.03 },
  { "x": "co2_per_unit_energy", "y": "energy_per_capita", "value": -0.08 },
  { "x": "co2_per_unit_energy", "y": "flaring_co2_per_capita", "value": 0.49 },
  { "x": "co2_per_unit_energy", "y": "nitrous_oxide_per_capita", "value": -0.01 },
  { "x": "co2_per_unit_energy", "y": "temperature_change_from_n2o", "value": -0.01 },
  { "x": "co2_per_unit_energy", "y": "co2", "value": -0.0 },
  { "x": "coal_co2_per_capita", "y": "population", "value": 0.07 },
  { "x": "coal_co2_per_capita", "y": "gdp", "value": 0.18 },
  { "x": "coal_co2_per_capita", "y": "cement_co2_per_capita", "value": 0.28 },
  { "x": "coal_co2_per_capita", "y": "co2_growth_abs", "value": 0.07 },
  { "x": "coal_co2_per_capita", "y": "co2_including_luc_growth_abs", "value": 0.04 },
  { "x": "coal_co2_per_capita", "y": "co2_including_luc_per_gdp", "value": -0.09 },
  { "x": "coal_co2_per_capita", "y": "co2_including_luc_per_unit_energy", "value": -0.05 },
  { "x": "coal_co2_per_capita", "y": "co2_per_gdp", "value": 0.22 },
  { "x": "coal_co2_per_capita", "y": "co2_per_unit_energy", "value": 0.03 },
  { "x": "coal_co2_per_capita", "y": "coal_co2_per_capita", "value": 1.0 },
  { "x": "coal_co2_per_capita", "y": "energy_per_capita", "value": 0.33 },
  { "x": "coal_co2_per_capita", "y": "flaring_co2_per_capita", "value": -0.03 },
  { "x": "coal_co2_per_capita", "y": "nitrous_oxide_per_capita", "value": 0.19 },
  { "x": "coal_co2_per_capita", "y": "temperature_change_from_n2o", "value": 0.21 },
  { "x": "coal_co2_per_capita", "y": "co2", "value": 0.23 },
  { "x": "energy_per_capita", "y": "population", "value": -0.02 },
  { "x": "energy_per_capita", "y": "gdp", "value": 0.15 },
  { "x": "energy_per_capita", "y": "cement_co2_per_capita", "value": 0.47 },
  { "x": "energy_per_capita", "y": "co2_growth_abs", "value": 0.03 },
  { "x": "energy_per_capita", "y": "co2_including_luc_growth_abs", "value": 0.01 },
  { "x": "energy_per_capita", "y": "co2_including_luc_per_gdp", "value": -0.12 },
  { "x": "energy_per_capita", "y": "co2_including_luc_per_unit_energy", "value": 0.0 },
  { "x": "energy_per_capita", "y": "co2_per_gdp", "value": 0.21 },
  { "x": "energy_per_capita", "y": "co2_per_unit_energy", "value": -0.08 },
  { "x": "energy_per_capita", "y": "coal_co2_per_capita", "value": 0.33 },
  { "x": "energy_per_capita", "y": "energy_per_capita", "value": 1.0 },
  { "x": "energy_per_capita", "y": "flaring_co2_per_capita", "value": 0.14 },
  { "x": "energy_per_capita", "y": "nitrous_oxide_per_capita", "value": 0.14 },
  { "x": "energy_per_capita", "y": "temperature_change_from_n2o", "value": 0.16 },
  { "x": "energy_per_capita", "y": "co2", "value": 0.17 },
  { "x": "flaring_co2_per_capita", "y": "population", "value": -0.02 },
  { "x": "flaring_co2_per_capita", "y": "gdp", "value": -0.02 },
  { "x": "flaring_co2_per_capita", "y": "cement_co2_per_capita", "value": 0.01 },
  { "x": "flaring_co2_per_capita", "y": "co2_growth_abs", "value": 0.0 },
  { "x": "flaring_co2_per_capita", "y": "co2_including_luc_growth_abs", "value": 0.0 },
  { "x": "flaring_co2_per_capita", "y": "co2_including_luc_per_gdp", "value": -0.01 },
  { "x": "flaring_co2_per_capita", "y": "co2_including_luc_per_unit_energy", "value": -0.01 },
  { "x": "flaring_co2_per_capita", "y": "co2_per_gdp", "value": 0.07 },
  { "x": "flaring_co2_per_capita", "y": "co2_per_unit_energy", "value": 0.49 },
  { "x": "flaring_co2_per_capita", "y": "coal_co2_per_capita", "value": -0.03 },
  { "x": "flaring_co2_per_capita", "y": "energy_per_capita", "value": 0.14 },
  { "x": "flaring_co2_per_capita", "y": "flaring_co2_per_capita", "value": 1.0 },
  { "x": "flaring_co2_per_capita", "y": "nitrous_oxide_per_capita", "value": 0.04 },
  { "x": "flaring_co2_per_capita", "y": "temperature_change_from_n2o", "value": -0.02 },
  { "x": "flaring_co2_per_capita", "y": "co2", "value": -0.01 },
  { "x": "nitrous_oxide_per_capita", "y": "population", "value": -0.05 },
  { "x": "nitrous_oxide_per_capita", "y": "gdp", "value": -0.0 },
  { "x": "nitrous_oxide_per_capita", "y": "cement_co2_per_capita", "value": 0.05 },
  { "x": "nitrous_oxide_per_capita", "y": "co2_growth_abs", "value": -0.01 },
  { "x": "nitrous_oxide_per_capita", "y": "co2_including_luc_growth_abs", "value": -0.0 },
  { "x": "nitrous_oxide_per_capita", "y": "co2_including_luc_per_gdp", "value": 0.22 },
  { "x": "nitrous_oxide_per_capita", "y": "co2_including_luc_per_unit_energy", "value": -0.03 },
  { "x": "nitrous_oxide_per_capita", "y": "co2_per_gdp", "value": 0.06 },
  { "x": "nitrous_oxide_per_capita", "y": "co2_per_unit_energy", "value": -0.01 },
  { "x": "nitrous_oxide_per_capita", "y": "coal_co2_per_capita", "value": 0.19 },
  { "x": "nitrous_oxide_per_capita", "y": "energy_per_capita", "value": 0.14 },
  { "x": "nitrous_oxide_per_capita", "y": "flaring_co2_per_capita", "value": 0.04 },
  { "x": "nitrous_oxide_per_capita", "y": "nitrous_oxide_per_capita", "value": 1.0 },
  { "x": "nitrous_oxide_per_capita", "y": "temperature_change_from_n2o", "value": 0.06 },
  { "x": "nitrous_oxide_per_capita", "y": "co2", "value": 0.01 },
  { "x": "temperature_change_from_n2o", "y": "population", "value": 0.68 },
  { "x": "temperature_change_from_n2o", "y": "gdp", "value": 0.89 },
  { "x": "temperature_change_from_n2o", "y": "cement_co2_per_capita", "value": 0.11 },
  { "x": "temperature_change_from_n2o", "y": "co2_growth_abs", "value": 0.31 },
  { "x": "temperature_change_from_n2o", "y": "co2_including_luc_growth_abs", "value": 0.15 },
  { "x": "temperature_change_from_n2o", "y": "co2_including_luc_per_gdp", "value": -0.07 },
  { "x": "temperature_change_from_n2o", "y": "co2_including_luc_per_unit_energy", "value": -0.02 },
  { "x": "temperature_change_from_n2o", "y": "co2_per_gdp", "value": 0.02 },
  { "x": "temperature_change_from_n2o", "y": "co2_per_unit_energy", "value": -0.01 },
  { "x": "temperature_change_from_n2o", "y": "coal_co2_per_capita", "value": 0.21 },
  { "x": "temperature_change_from_n2o", "y": "energy_per_capita", "value": 0.16 },
  { "x": "temperature_change_from_n2o", "y": "flaring_co2_per_capita", "value": -0.02 },
  { "x": "temperature_change_from_n2o", "y": "nitrous_oxide_per_capita", "value": 0.06 },
  { "x": "temperature_change_from_n2o", "y": "temperature_change_from_n2o", "value": 1.0 },
  { "x": "temperature_change_from_n2o", "y": "co2", "value": 0.86 },
  { "x": "co2", "y": "population", "value": 0.66 },
  { "x": "co2", "y": "gdp", "value": 0.93 },
  { "x": "co2", "y": "cement_co2_per_capita", "value": 0.18 },
  { "x": "co2", "y": "co2_growth_abs", "value": 0.46 },
  { "x": "co2", "y": "co2_including_luc_growth_abs", "value": 0.23 },
  { "x": "co2", "y": "co2_including_luc_per_gdp", "value": -0.06 },
  { "x": "co2", "y": "co2_including_luc_per_unit_energy", "value": -0.02 },
  { "x": "co2", "y": "co2_per_gdp", "value": 0.05 },
  { "x": "co2", "y": "co2_per_unit_energy", "value": -0.0 },
  { "x": "co2", "y": "coal_co2_per_capita", "value": 0.23 },
  { "x": "co2", "y": "energy_per_capita", "value": 0.17 },
  { "x": "co2", "y": "flaring_co2_per_capita", "value": -0.01 },
  { "x": "co2", "y": "nitrous_oxide_per_capita", "value": 0.01 },
  { "x": "co2", "y": "temperature_change_from_n2o", "value": 0.86 },
  { "x": "co2", "y": "co2", "value": 1.0 }
];

export const getCorrelationsWithCO2 = (): CorrelationPoint[] => {
  return correlationData.filter(point => point.y === "co2" && point.x !== "co2");
};

export interface FeatureInsight {
  title: string;
  description: string;
}

export const getFeatureInsights = (): FeatureInsight[] => {
  return [
    {
      title: "GDP & Temperature Are Key Drivers",
      description: "GDP (0.93) and temperature change from N₂O (0.86) show the strongest correlations with CO₂ emissions, highlighting economic activity and climate feedback as primary factors."
    },
    {
      title: "Population Growth Matters",
      description: "Population (0.66) has a strong correlation with emissions - more people generally means higher energy consumption and more emissions."
    },
    {
      title: "Trend Indicators Are Important",
      description: "CO₂ growth metrics (0.46, 0.23) are significant predictors, capturing the momentum and direction of emission changes."
    },
    {
      title: "Energy Efficiency Shows Complex Relationships",
      description: "Efficiency metrics like CO₂ per GDP and CO₂ per unit energy have lower correlations, showing that improving efficiency can moderate emissions growth."
    },
    {
      title: "Fossil Fuel Dependence",
      description: "Coal CO₂ per capita (0.23) remains a significant factor, highlighting the persistent impact of fossil fuel choices on overall emissions."
    }
  ];
};
