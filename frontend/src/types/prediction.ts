export interface PredictionPayload {
    year: number;
    population: number;
    gdp: number;
    cement_co2_per_capita: number;
    co2_growth_abs: number;
    co2_including_luc_growth_abs: number;
    co2_including_luc_per_gdp: number;
    co2_including_luc_per_unit_energy: number;
    co2_per_gdp: number;
    co2_per_unit_energy: number;
    coal_co2_per_capita: number;
    energy_per_capita: number;
    flaring_co2_per_capita: number;
    nitrous_oxide_per_capita: number;
    temperature_change_from_n2o: number;
  }
  
  export interface PredictionResponse {
    prediction: number;
  }
  
<<<<<<< HEAD
  export interface SliderConfig {
    id: keyof PredictionPayload;
    label: string;
    min: number;
    max: number;
    step: number;
    unit?: string;
    initialValue: number;
  }
=======
  export interface InputConfig {
    id: keyof Omit<PredictionPayload, 'year'>;
    label: string;
    unit?: string;
    initialValue: number;
  }
  
>>>>>>> 84a8efb (refactored version)
