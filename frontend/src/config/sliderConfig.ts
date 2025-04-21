
import { SliderConfig } from "@/types/prediction";

export const BASELINE_2023 = 264.4;

export const sliderConfigs: SliderConfig[] = [
  {
    id: "year",
    label: "Forecast Year",
    min: 2023,
    max: 2030,
    step: 1,
    initialValue: 2023
  },
  {
    id: "population",
    label: "Population",
    min: 65,
    max: 80,
    step: 0.1,
    unit: "Million",
    initialValue: 71.7
  },
  {
    id: "gdp",
    label: "GDP",
    min: 10,
    max: 20,
    step: 0.01,
    unit: "Trillion ฿",
    initialValue: 11.24
  },
  {
    id: "cement_co2_per_capita",
    label: "Cement CO₂ per Capita",
    min: 0.1,
    max: 0.5,
    step: 0.001,
    unit: "t",
    initialValue: 0.293
  },
  {
    id: "co2_growth_abs",
    label: "CO₂ Growth (Absolute)",
    min: -10,
    max: 20,
    step: 0.001,
    unit: "Mt",
    initialValue: 5.432
  },
  {
    id: "co2_including_luc_growth_abs",
    label: "CO₂ incl. LUC Growth",
    min: -10,
    max: 20,
    step: 0.001,
    unit: "Mt",
    initialValue: 4.502
  },
  {
    id: "co2_including_luc_per_gdp",
    label: "CO₂ incl. LUC per GDP",
    min: 0.1,
    max: 0.5,
    step: 0.001,
    unit: "kg/$",
    initialValue: 0.273
  },
  {
    id: "co2_including_luc_per_unit_energy",
    label: "CO₂ incl. LUC per Energy",
    min: 0.1,
    max: 0.5,
    step: 0.001,
    unit: "kg/kWh",
    initialValue: 0.221
  },
  {
    id: "co2_per_gdp",
    label: "CO₂ per GDP",
    min: 0.1,
    max: 0.5,
    step: 0.001,
    unit: "kg/$",
    initialValue: 0.242
  },
  {
    id: "co2_per_unit_energy",
    label: "CO₂ per Unit Energy",
    min: 0.1,
    max: 0.5,
    step: 0.001,
    unit: "kg/kWh",
    initialValue: 0.196
  },
  {
    id: "coal_co2_per_capita",
    label: "Coal CO₂ per Capita",
    min: 0.5,
    max: 2,
    step: 0.001,
    unit: "t",
    initialValue: 0.973
  },
  {
    id: "energy_per_capita",
    label: "Energy per Capita",
    min: 15000,
    max: 25000,
    step: 0.1,
    unit: "kWh",
    initialValue: 19357.754
  },
  {
    id: "flaring_co2_per_capita",
    label: "Flaring CO₂ per Capita",
    min: 0,
    max: 0.02,
    step: 0.001,
    unit: "t",
    initialValue: 0.006
  },
  {
    id: "nitrous_oxide_per_capita",
    label: "N₂O per Capita",
    min: 0.1,
    max: 1,
    step: 0.001,
    unit: "t",
    initialValue: 0.339
  },
  {
    id: "temperature_change_from_n2o",
    label: "Temp Change from N₂O",
    min: 0,
    max: 0.01,
    step: 0.001,
    unit: "°C",
    initialValue: 0.001
  }
];
