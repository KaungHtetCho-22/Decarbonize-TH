import { InputConfig } from "@/types/prediction";

export const BASELINE_2023 = 264.4;

export const inputConfigs: InputConfig[] = [
  {
    id: "population",
    label: "Population",
    unit: "Million",
    initialValue: 71.7
  },
  {
    id: "gdp",
    label: "GDP",
    unit: "Trillion ฿",
    initialValue: 11.24
  },
  {
    id: "cement_co2_per_capita",
    label: "Cement CO₂ per Capita",
    unit: "t",
    initialValue: 0.293
  },
  {
    id: "co2_growth_abs",
    label: "CO₂ Growth (Absolute)",
    unit: "Mt",
    initialValue: 5.432
  },
  {
    id: "co2_including_luc_growth_abs",
    label: "CO₂ incl. LUC Growth",
    unit: "Mt",
    initialValue: 4.502
  },
  {
    id: "co2_including_luc_per_gdp",
    label: "CO₂ incl. LUC per GDP",
    unit: "kg/$",
    initialValue: 0.273
  },
  {
    id: "co2_including_luc_per_unit_energy",
    label: "CO₂ incl. LUC per Energy",
    unit: "kg/kWh",
    initialValue: 0.221
  },
  {
    id: "co2_per_gdp",
    label: "CO₂ per GDP",
    unit: "kg/$",
    initialValue: 0.242
  },
  {
    id: "co2_per_unit_energy",
    label: "CO₂ per Unit Energy",
    unit: "kg/kWh",
    initialValue: 0.196
  },
  {
    id: "coal_co2_per_capita",
    label: "Coal CO₂ per Capita",
    unit: "t",
    initialValue: 0.973
  },
  {
    id: "energy_per_capita",
    label: "Energy per Capita",
    unit: "kWh",
    initialValue: 19357.754
  },
  {
    id: "flaring_co2_per_capita",
    label: "Flaring CO₂ per Capita",
    unit: "t",
    initialValue: 0.006
  },
  {
    id: "nitrous_oxide_per_capita",
    label: "N₂O per Capita",
    unit: "t",
    initialValue: 0.339
  },
  {
    id: "temperature_change_from_n2o",
    label: "Temp Change from N₂O",
    unit: "°C",
    initialValue: 0.001
  }
];
