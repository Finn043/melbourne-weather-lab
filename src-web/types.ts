export interface WeatherDay {
  date: string;
  tmin: number;
  tmax: number;
  prcp: number;
}

export interface WeatherDataset {
  station: string;
  startYear: number;
  endYear: number;
  records: WeatherDay[];
}

export interface LstmLayer {
  inputSize: number;
  units: number;
  kernel: number[][];
  recurrent: number[][];
  bias: number[];
}

export interface DenseLayer {
  kernel: number[][];
  bias: number[];
  activation: 'relu' | 'linear';
}

export interface WeatherGroup {
  id: number;
  center: number[];
  label: string;
  profile: Pick<WeatherDay, 'tmin' | 'tmax' | 'prcp'>;
}

export interface BrowserModel {
  features: string[];
  lookback: number;
  featureMean: number[];
  featureScale: number[];
  targetScale: number[];
  targetMin: number[];
  lstm: LstmLayer[];
  dense: DenseLayer[];
  regression: { coefficients: number[]; intercept: number };
  clusters: { mean: number[]; scale: number[]; groups: WeatherGroup[] };
}
